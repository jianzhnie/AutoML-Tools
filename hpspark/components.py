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
    """
    LogisticRegression(*args,
            featuresCol="features", labelCol="label", 
            predictionCol="prediction", maxIter=100, 
            regParam=0.0, elasticNetParam=0.0, tol=1e-6, 
            fitIntercept=True, threshold=0.5, thresholds=None, 
            probabilityCol="probability", rawPredictionCol="rawPrediction", 
            standardization=True, weightCol=None, aggregationDepth=2, family="auto", 
            lowerBoundsOnCoefficients=None, upperBoundsOnCoefficients=None, lowerBoundsOnIntercepts=None, 
            upperBoundsOnIntercepts=None)
    """  
    return LogisticRegression(*args, **kwargs)


@scope.define
def spark_DecisionTreeClassifier(*args, **kwargs):
    """
    DecisionTreeClassifier(*args, 
            featuresCol="features", labelCol="label", 
            predictionCol="prediction", probabilityCol="probability", 
            rawPredictionCol="rawPrediction", maxDepth=5, maxBins=32, 
            minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
            cacheNodeIds=False, checkpointInterval=10, impurity="gini", 
            seed=None)
    """
    return DecisionTreeClassifier(*args, **kwargs)



@scope.define
def spark_RandomForestClassifier(*args, **kwargs):
    """
    RandomForestClassifier(*args, 
            featuresCol="features", labelCol="label", 
            predictionCol="prediction", probabilityCol="probability", 
            rawPredictionCol="rawPrediction", maxDepth=5, maxBins=32, 
            minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
            cacheNodeIds=False, checkpointInterval=10, impurity="gini", 
            numTrees=20, featureSubsetStrategy="auto", seed=None, 
            subsamplingRate=1.0)
    """
    return RandomForestClassifier(*args, **kwargs)


@scope.define
def spark_GBTClassifier(*args, **kwargs):
    """
    GBTClassifier(*args, 
            featuresCol="features", labelCol="label", 
            predictionCol="prediction", maxDepth=5, maxBins=32, 
            minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
            cacheNodeIds=False, checkpointInterval=10, lossType="logistic", 
            maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0, 
            featureSubsetStrategy="all")
    """
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

def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])


def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])

def _grad_boosting_reg_loss_alpha(name):
    return hp.choice(name, [
        ('ls', 0.9),
        ('lad', 0.9),
        ('huber', hp.uniform(name + '.alpha', 0.85, 0.95)),
        ('quantile', 0.5)
    ])

def _grad_boosting_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _grad_boosting_subsample(name):
    return hp.pchoice(name, [
        (0.2, 1.0),  # default choice.
        (0.8, hp.uniform(name + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
    ])


####################################################################
##==== Random forest hyperparameters search space ====##
####################################################################
def _trees_hp_space(
        name_func,
        subsamplingRate=None,
        numTrees=None,
        maxDepth=None,
        minInstancesPerNode=None
        ):
    '''Generate trees ensemble hyperparameters search space
    '''
    hp_space = dict(
        subsamplingRate=(_grad_boosting_subsample(name_func('subsamplingRate'))
                            if subsamplingRate is None else subsamplingRate),
        numTrees=(_trees_n_estimators(name_func('numTrees'))
                      if numTrees is None else numTrees),
        maxDepth=(_trees_max_depth(name_func('maxDepth'))
                   if maxDepth is None else maxDepth),
        minInstancesPerNode=(_trees_min_samples_leaf(name_func('minInstancesPerNode'))
                          if minInstancesPerNode is None else minInstancesPerNode),
    )
    return hp_space

      
#############################################################
##==== Random forest classifier constructors ====##
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
    return scope.spark_RandomForestClassifier(**hp_space)


###########################################################
##==== GradientBoosting hyperparameters search space ====##
###########################################################
def _grad_boosting_hp_space(
    name_func,
    stepSize=None,
    subsamplingRate=None,
    numTrees=None,
    maxDepth=None,
    minInstancesPerNode=None
    ):
    '''Generate GradientBoosting hyperparameters search space
    '''
    hp_space = dict(
        stepSize=(_grad_boosting_learning_rate(name_func('stepSize'))
                       if stepSize is None else stepSize),
        subsamplingRate=(_grad_boosting_subsample(name_func('subsamplingRate'))
                            if subsamplingRate is None else subsamplingRate),
        numTrees=(_trees_n_estimators(name_func('numTrees'))
                      if numTrees is None else numTrees),
        maxDepth=(_trees_max_depth(name_func('maxDepth'))
                   if maxDepth is None else maxDepth),
        minInstancesPerNode=(_trees_min_samples_leaf(name_func('minInstancesPerNode'))
                          if minInstancesPerNode is None else minInstancesPerNode),
    )
    return hp_space


################################################################
##==== GradientBoosting classifier constructors ====##
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

    return scope.spark_GBTClassifier(**hp_space)


###########################################################
##==== GradientBoosting hyperparameters search space ====##
###########################################################

def _decision_tree_hp_space(
    name_func,
    maxDepth=None,
    minInstancesPerNode=None
    ):
    '''Generate decision_tree hyperparameters search space
    '''
    hp_space = dict(
        maxDepth=(_trees_max_depth(name_func('maxDepth'))
                   if maxDepth is None else maxDepth),
        minInstancesPerNode=(_trees_min_samples_leaf(name_func('minInstancesPerNode'))
                          if minInstancesPerNode is None else minInstancesPerNode),
    )
    return hp_space


##################################################
##==== Decision tree classifier constructor ====##
##################################################

def decision_tree(name, loss=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.GradientBoostingClassifier model.

    Args:
        loss([str]): choose from ['deviance', 'exponential']

    See help(hpsklearn.components._grad_boosting_hp_space) for info on
    additional available GradientBoosting arguments.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'decision_tree', msg)

    hp_space = _decision_tree_hp_space(_name, **kwargs)

    return scope.spark_GBTClassifier(**hp_space)


####################################################
##==== Various classifier/regressor selectors ====##
####################################################
def any_classifier(name):
    classifiers = [
        decision_tree(name + '.decision_tree'),
        random_forest(name + '.random_forest'),
        gradient_boosting(name + '.grad_boosting', loss='deviance'),
    ]
    return hp.choice('%s' % name, classifiers)