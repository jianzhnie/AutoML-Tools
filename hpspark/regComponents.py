import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from functools import partial
from hyperopt.pyll import scope, as_apply
from hyperopt import hp




@scope.define
def spark_LogisticRegression(*args, **kwargs):
    return LogisticRegression(*args, **kwargs)


@scope.define
def spark_DecisionTreeClassifier(*args, **kwargs):
    return DecisionTreeClassifier(*args, **kwargs)


@scope.define
def spark_RandomForestClassifier(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs)


@scope.define
def spark_GBTClassifier(*args, **kwargs):
    return GBTClassifier(*args, **kwargs)


@scope.define
def patience_param(x):
    """
    Mark a hyperparameter as having a simple monotonic increasing
    relationship with both CPU time and the goodness of the model.
    """
    # -- TODO: make this do something!
    return x

@scope.define
def inv_patience_param(x):
    """
    Mark a hyperparameter as having a simple monotonic decreasing
    relationship with both CPU time and the goodness of the model.
    """
    # -- TODO: make this do something!
    return x


def _trees_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))

def _trees_criterion(name):
    return hp.choice(name, ['gini', 'entropy'])

def _trees_max_features(name):
    return hp.pchoice(name, [
        (0.2, 'sqrt'),  # most common choice.
        (0.1, 'log2'),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + '.frac', 0., 1.))
    ])

def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])


def _trees_min_samples_split(name):
    return 2

def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])

def _trees_bootstrap(name):
    return hp.choice(name, [True, False])

def _boosting_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1))

def _ada_boost_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _ada_boost_loss(name):
    return hp.choice(name, ['linear', 'square', 'exponential'])

def _ada_boost_algo(name):
    return hp.choice(name, ['SAMME', 'SAMME.R'])

def _grad_boosting_reg_loss_alpha(name):
    return hp.choice(name, [
        ('ls', 0.9),
        ('lad', 0.9),
        ('huber', hp.uniform(name + '.alpha', 0.85, 0.95)),
        ('quantile', 0.5)
    ])

def _grad_boosting_clf_loss(name):
    return hp.choice(name, ['deviance', 'exponential'])

def _grad_boosting_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _grad_boosting_subsample(name):
    return hp.pchoice(name, [
        (0.2, 1.0),  # default choice.
        (0.8, hp.uniform(name + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
    ])


def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state

def _class_weight(name):
    return hp.choice(name, [None, 'balanced'])


####################################################################
##==== Random forest/extra trees hyperparameters search space ====##
####################################################################
def _trees_hp_space(
        name_func,
        n_estimators=None,
        max_features=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=False):
    '''Generate trees ensemble hyperparameters search space
    '''
    hp_space = dict(
        n_estimators=(_trees_n_estimators(name_func('n_estimators'))
                      if n_estimators is None else n_estimators),
        max_features=(_trees_max_features(name_func('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(name_func('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(name_func('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(name_func('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(name_func('rstate'), random_state),
        verbose=verbose,
    )
    return hp_space



#############################################################
##==== Random forest classifier/regressor constructors ====##
#############################################################
def random_forest(name, criterion=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.RandomForestClassifier model.

    Args:
        criterion([str]): choose 'gini' or 'entropy'.

    See help(hpsklearn.components._trees_hp_space) for info on additional
    available random forest/extra trees arguments.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'rfc', msg)

    hp_space = _trees_hp_space(_name, **kwargs)
    hp_space['criterion'] = (_trees_criterion(_name('criterion'))
                             if criterion is None else criterion)
    return scope.spark_RandomForestClassifier(**hp_space)


###########################################################
##==== GradientBoosting hyperparameters search space ====##
###########################################################
def _grad_boosting_hp_space(
    name_func,
    learning_rate=None,
    n_estimators=None,
    subsample=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_depth=None,
    init=None,
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False):
    '''Generate GradientBoosting hyperparameters search space
    '''
    hp_space = dict(
        learning_rate=(_grad_boosting_learning_rate(name_func('learning_rate'))
                       if learning_rate is None else learning_rate),
        n_estimators=(_boosting_n_estimators(name_func('n_estimators'))
                      if n_estimators is None else n_estimators),
        subsample=(_grad_boosting_subsample(name_func('subsample'))
                   if subsample is None else subsample),
        min_samples_split=(_trees_min_samples_split(name_func('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(name_func('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        max_depth=(_trees_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        init=init,
        random_state=_random_state(name_func('rstate'), random_state),
        max_features=(_trees_max_features(name_func('max_features'))
                   if max_features is None else max_features),
        warm_start=warm_start,
    )
    return hp_space


################################################################
##==== GradientBoosting classifier/regressor constructors ====##
################################################################
def gradient_boosting(name, loss=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.GradientBoostingClassifier model.

    Args:
        loss([str]): choose from ['deviance', 'exponential']

    See help(hpsklearn.components._grad_boosting_hp_space) for info on
    additional available GradientBoosting arguments.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'gradient_boosting', msg)

    hp_space = _grad_boosting_hp_space(_name, **kwargs)
    hp_space['loss'] = (_grad_boosting_clf_loss(_name('loss'))
                        if loss is None else loss)
    return scope.sklearn_GradientBoostingClassifier(**hp_space)


def gradient_boosting_regression(name, loss=None, alpha=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.GradientBoostingRegressor model.

    Args:
        loss([str]): choose from ['ls', 'lad', 'huber', 'quantile']
        alpha([float]): alpha parameter for huber and quantile losses.
                        Must be within [0.0, 1.0].

    See help(hpsklearn.components._grad_boosting_hp_space) for info on
    additional available GradientBoosting arguments.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'gradient_boosting_reg', msg)

    loss_alpha = _grad_boosting_reg_loss_alpha(_name('loss_alpha'))
    hp_space = _grad_boosting_hp_space(_name, **kwargs)
    hp_space['loss'] = loss_alpha[0] if loss is None else loss
    hp_space['alpha'] = loss_alpha[1] if alpha is None else alpha
    return scope.sklearn_GradientBoostingRegressor(**hp_space)
