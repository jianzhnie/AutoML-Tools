# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
import copy
from functools import partial
from multiprocessing import Process, Pipe
import time
import inspect
import numpy as np
import hyperopt
from pyspark.ml import Estimator
from components import any_classifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator



# Constants for partial_fit

# The partial_fit method will not be run if there is less than
# timeout * timeout_buffer number of seconds left before timeout
timeout_buffer = 0.05

# The minimum number of iterations of the partial_fit method that must be run
# before early stopping can kick in is min_n_iters
min_n_iters = 7

# After best_loss_cutoff_n_iters iterations have occured, the training can be
# stopped early if the validation scores are far from the best scores
best_loss_cutoff_n_iters = 35

# Early stopping can occur when the best validation score of the earlier runs is
# greater than that of the later runs, tipping_pt_ratio determines the split
tipping_pt_ratio = 0.6

# Retraining will be done with all training data for retrain_fraction
# multiplied by the number of iterations used to train the original learner
retrain_fraction = 1.2


def _cost_fn(argd, X, y, valid_size, n_folds, shuffle, random_state,
             use_partial_fit, info, timeout, _conn, loss_fn=None,
             continuous_loss_fn=False, best_loss=None, n_jobs=1):
    '''Calculate the loss function
    '''
    t_start = time.time()
    # Extract info from calling function.
    if 'classifier' in argd:
        classifier = argd['classifier']
        regressor = argd['regressor']
    else:
        classifier = argd['model']['classifier']
        regressor = argd['model']['regressor']
    learner = classifier if classifier is not None else regressor

    is_classif = classifier is not None
    untrained_learner = copy.deepcopy(learner)

    if loss_fn is None:
        if is_classif:
            loss = 1 - accuracy_score(test_dataset)
            # -- squared standard error of mean
            lossvar = (loss * (1 - loss)) / max(1, len(cv_y_pool) - 1)
            info('OK trial with accuracy %.1f +- %.1f' % (
                    100 * (1 - loss),
                    100 * np.sqrt(lossvar))
            )
        else:
            loss = 1 - r2_score(cv_y_pool, cv_pred_pool)
            lossvar = None  # variance of R2 is undefined.
            info('OK trial with R2 score %.2e' % (1 - loss))
    else:
        # Use a user specified loss function
        loss = loss_fn(cv_y_pool, cv_pred_pool)
        lossvar = None
        info('OK trial with loss %.1f' % loss)
    t_done = time.time()
    rval = {
        'loss': loss,
        'loss_variance': lossvar,
        'learner': untrained_learner,
        'status': hyperopt.STATUS_OK,
        'duration': t_done - t_start,
        'iterations': (cv_n_iters.max()
            if (hasattr(learner, "partial_fit") and use_partial_fit)
            else None),
    }
    rtype = 'return'
    # -- return the result to calling process
    _conn.send((rtype, rval))


class HpSparkEstimator(Estimator):
    """Automatically creates and optimizes machine learning pipelines using GP."""

    def __init__(self,
                 classifier=None,
                 regressor=None,
                 space=None,
                 algo=None,
                 max_evals=10,
                 loss_fn=None,
                 continuous_loss_fn=False,
                 verbose=False,
                 trial_timeout=None,
                 fit_increment=1,
                 fit_increment_dump_filename=None,
                 seed=None,
                 n_jobs=1
                 ):
        """
        Parameters
        ----------

        classifier: pyll.Apply node
            This should evaluates to sklearn-style classifier (may include
            hyperparameters).

        regressor: pyll.Apply node
            This should evaluates to sklearn-style regressor (may include
            hyperparameters).

        algo: hyperopt suggest algo (e.g. rand.suggest)

        max_evals: int
            Fit() will evaluate up to this-many configurations. Does not apply
            to fit_iter, which continues to search indefinitely.

        loss_fn: callable
            A function that takes the arguments (y_target, y_prediction)
            and computes a loss value to be minimized. If no function is
            specified, '1.0 - accuracy_score(y_target, y_prediction)' is used
            for classification and '1.0 - r2_score(y_target, y_prediction)'
            is used for regression

        continuous_loss_fn: boolean, default is False
            When true, the loss function is passed the output of
            predict_proba() as the second argument.  This is to facilitate the
            use of continuous loss functions like cross entropy or AUC.  When
            false, the loss function is given the output of predict().  If
            true, `classifier` and `loss_fn` must also be specified.

        trial_timeout: float (seconds), or None for no timeout
            Kill trial evaluations after this many seconds.

        fit_increment: int
            Every this-many trials will be a synchronization barrier for
            ongoing trials, and the hyperopt Trials object may be
            check-pointed.  (Currently evaluations are done serially, but
            that might easily change in future to allow e.g. MongoTrials)

        fit_increment_dump_filename : str or None
            Periodically dump self.trials to this file (via cPickle) during
            fit()  Saves after every `fit_increment` trial evaluations.

        seed: numpy.random.RandomState or int or None
            If int, the integer will be used to seed a RandomState instance
            for use in hyperopt.fmin. Use None to make sure each run is
            independent. Default is None.

        n_jobs: integer, default 1
            Use multiple CPU cores when training estimators which support
            multiprocessing.
        """
        self.max_evals = max_evals
        self.loss_fn = loss_fn
        self.continuous_loss_fn = continuous_loss_fn
        self.verbose = verbose
        self.trial_timeout = trial_timeout
        self.fit_increment = fit_increment
        self.fit_increment_dump_filename = fit_increment_dump_filename
        self._best_preprocs = ()
        self._best_ex_preprocs = ()
        self._best_learner = None
        self._best_loss = None
        self._best_iters = None
        self.n_jobs = n_jobs
        if space is None:
            if classifier is None and regressor is None:
                self.classification = True
                classifier = any_classifier('classifier')
            elif classifier is not None:
                assert regressor is None
                self.classification = True
            else:
                assert regressor is not None
                self.classification = False
            self.space = hyperopt.pyll.as_apply({
                'classifier': classifier,
                'regressor': regressor,
            })
        else:
            assert classifier is None
            assert regressor is None
            # self.space = hyperopt.pyll.as_apply(space)
            self.space = space
            evaled_space = space.eval()
            if 'ex_preprocs' in evaled_space:
                self.n_ex_pps = len(evaled_space['ex_preprocs'])
            else:
                self.n_ex_pps = 0
                self.ex_preprocs = []

        if algo is None:
            self.algo = hyperopt.rand.suggest
        else:
            self.algo = algo

        if seed is not None:
            self.rstate = (np.random.RandomState(seed)
                           if isinstance(seed, int) else seed)
        else:
            self.rstate = np.random.RandomState()

        # Backwards compatibility with older version of hyperopt
        self.seed = seed
        if 'rstate' not in inspect.getargspec(hyperopt.fmin).args:
            print("Warning: Using older version of hyperopt.fmin")

        if self.continuous_loss_fn:
            assert self.space['classifier'] is not None, \
                "Can only use continuous_loss_fn with classifiers."
            assert self.loss_fn is not None, \
                "Must specify loss_fn if continuous_loss_fn is true."

    def info(self, *args):
        if self.verbose:
            print(' '.join(map(str, args)))


    def fit_iter(self, dataset, valid_size=.2, n_folds=None,
                 cv_shuffle=False, warm_start=False,
                 random_state=np.random.RandomState(),
                 weights=None, increment=None):
        """Generator of Trials after ever-increasing numbers of evaluations
        """
        assert weights is None
        increment = self.fit_increment if increment is None else increment

        if not warm_start:
            self.trials = hyperopt.Trials()
            self._best_loss = float('inf')
        else:
            assert hasattr(self, 'trials')
        # self._best_loss = float('inf')
        # This is where the cost function is used.
        fn = partial(_cost_fn,
                     dataset,
                     valid_size=valid_size, n_folds=n_folds,
                     shuffle=cv_shuffle, random_state=random_state,
                     use_partial_fit=self.use_partial_fit,
                     info=self.info,
                     timeout=self.trial_timeout,
                     loss_fn=self.loss_fn,
                     continuous_loss_fn=self.continuous_loss_fn,
                     n_jobs=self.n_jobs)

        # Wrap up the cost function as a process with timeout control.
        def fn_with_timeout(*args, **kwargs):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=partial(fn, best_loss=self._best_loss),
                         args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(self.trial_timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                self.info('TERMINATING DUE TO TIMEOUT')
                th.terminate()
                th.join()
                fn_rval = 'return', {
                    'status': hyperopt.STATUS_FAIL,
                    'failure': 'TimeOut'
                }

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]

            # -- remove potentially large objects from the rval
            #    so that the Trials() object below stays small
            #    We can recompute them if necessary, and it's usually
            #    not necessary at all.
            if fn_rval[1]['status'] == hyperopt.STATUS_OK:
                fn_loss = float(fn_rval[1].get('loss'))
                fn_learner = fn_rval[1].pop('learner')
                fn_iters = fn_rval[1].pop('iterations')
                if fn_loss < self._best_loss:
                    self._best_learner = fn_learner
                    self._best_loss = fn_loss
                    self._best_iters = fn_iters
            return fn_rval[1]

        while True:
            new_increment = yield self.trials
            if new_increment is not None:
                increment = new_increment
            if 'rstate' in inspect.getargspec(hyperopt.fmin).args:
                hyperopt.fmin(fn_with_timeout,
                                space=self.space,
                                algo=self.algo,
                                trials=self.trials,
                                max_evals=len(self.trials.trials) + increment,
                                rstate=self.rstate,
                                # -- let exceptions crash the program,
                                #    so we notice them.
                                catch_eval_exceptions=False,
                                return_argmin=False, # -- in case no success so far
                                )
            else:
                if self.seed is None:
                    hyperopt.fmin(fn_with_timeout,
                                  space=self.space,
                                  algo=self.algo,
                                  trials=self.trials,
                                  max_evals=len(self.trials.trials) + increment,
                                 )

    def predict(self, dataset, fit_preproc=False):
        """
        Use the best model found by previous fit() to make a prediction.
        """
        if self._best_learner is None:
            raise RuntimeError(
                "Attempting to use a model that has not been fit. "
                "Ensure fit() has been called and at least one trial "
                "has completed without failing or timing out."
            )

        return self._best_learner.predict(dataset)


    def score(self, dataset, fit_preproc=False):
        """
        Return the score (accuracy or R2) of the learner on
        a given set of data
        """
        if self._best_learner is None:
            raise RuntimeError(
                "Attempting to use a model that has not been fit. "
                "Ensure fit() has been called and at least one trial "
                "has completed without failing or timing out."
            )
        return self._best_learner.score(dataset)


    def best_model(self):
        """
        Returns the best model found by the previous fit()
        """
        return {'learner': self._best_learner,
                'preprocs': self._best_preprocs,
                'ex_preprocs': self._best_ex_preprocs}

