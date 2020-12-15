"""
hp.choice   返回一个选项，选项可以是list或者tuple.options可以是嵌套的表达式，用于组成条件参数。 
hp.pchoice(label,p_options)以一定的概率返回一个p_options的一个选项。这个选项使得函数在搜索过程中对每个选项的可能性不均匀。 
hp.uniform(label,low,high)参数在low和high之间均匀分布。 
hp.quniform (label,low,high,q),参数的取值是round(uniform(low,high)/q)*q，适用于那些离散的取值。 
hp.loguniform(label,low,high)绘制exp(uniform(low,high)),变量的取值范围是[exp(low),exp(high)] 
hp.randint(label,upper) 返回一个在[0,upper)前闭后开的区间内的随机整数。 
"""

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope, as_apply
from pyspark.ml.regression import (LinearRegression, DecisionTreeRegressor, 
                GBTRegressor, RandomForestRegressor)
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, 
                GBTClassifier, DecisionTreeClassifier)

### Classification
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


### Rgression
@scope.define
def spark_LinearRegression(*args, **kwargs):
    """
    LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction", 
                    maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, 
                    standardization=True, solver="auto", weightCol=None, aggregationDepth=2, 
                    loss="squaredError", epsilon=1.35)

    The learning objective is to minimize the specified loss function, with regularization. This supports two kinds of loss:

    squaredError (a.k.a squared loss)

    huber (a hybrid of squared error for relatively small errors and absolute error for relatively large ones, and we estimate the scale parameter from training data)

    This supports multiple types of regularization:

    none (a.k.a. ordinary least squares)

    L2 (ridge regression)

    L1 (Lasso)

    L2 + L1 (elastic net)
    """
    return LinearRegression(*args, **kwargs)


@scope.define
def spark_DecisionTreeRegressor(*args, **kwargs):
    """
    DecisionTreeRegressor(*args, 
            featuresCol="features", labelCol="label", 
            predictionCol="prediction", maxDepth=5, maxBins=32, 
            minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
            cacheNodeIds=False, checkpointInterval=10, impurity="variance", 
            seed=None, varianceCol=None, weightCol=None, leafCol="", 
            minWeightFractionPerNode=0.0)
    """
    return DecisionTreeRegressor(*args, **kwargs)


@scope.define
def spark_RandomForestRegressor(*args, **kwargs):
    """
    RandomForestRegressor(*args, 
        featuresCol="features", labelCol="label", 
        predictionCol="prediction", maxDepth=5, maxBins=32,
         minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
         cacheNodeIds=False, checkpointInterval=10, impurity="variance", 
         subsamplingRate=1.0, seed=None, numTrees=20, featureSubsetStrategy="auto", 
         leafCol="", minWeightFractionPerNode=0.0, weightCol=None, bootstrap=True)
    """
    return RandomForestRegressor(*args, **kwargs)


@scope.define
def spark_GBTRegressor(*args, **kwargs):
    """
    GBTRegressor(*args, 
        featuresCol="features", labelCol="label", 
        predictionCol="prediction", maxDepth=5, maxBins=32, 
        minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, 
        cacheNodeIds=False, subsamplingRate=1.0, checkpointInterval=10, 
        lossType="squared", maxIter=20, stepSize=0.1, seed=None, impurity="variance", 
        featureSubsetStrategy="all", validationTol=0.01, validationIndicatorCol=None, 
        leafCol="", minWeightFractionPerNode=0.0, weightCol=None)    
    """
    return GBTRegressor(*args, **kwargs)



def _elastic_net_ratio(name):
    return hp.uniform(name, 0, 1)


def _regularization_penalty(name):
    return hp.loguniform(name, np.log(1e-6), np.log(1e-1))


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
    return hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))


def _grad_boosting_subsample(name):
    return hp.pchoice(name, [
        (0.2, 1.0),  # default choice.
        (0.8, hp.uniform(name + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
    ])



cls_space = hp.choice('classifier_type', [
    {
        'type': 'decision_tree',
        'maxDepth' : _trees_max_depth('decision_tree.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('decision_tree.minInstancesPerNode')
    },
    {
        'type': 'randomforest',
        'subsamplingRate': _grad_boosting_subsample('randomforest.subsamplingRate'),
        'numTrees' :  _trees_n_estimators('randomforest.numTrees'),
        'maxDepth': _trees_max_depth('randomforest.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('randomforest.minInstancesPerNode')
    },
    {
        'type': 'grad_boosting',
        'stepSize': _grad_boosting_learning_rate('grad_boosting.stepSize'),
        'subsamplingRate': _grad_boosting_subsample('grad_boosting.subsamplingRate'),
        'maxDepth': _trees_max_depth('grad_boosting.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('grad_boosting.minInstancesPerNode')
    }
])


reg_space = hp.choice('regresssior_type', [
    {
        'type': 'linear_regression',
        'regParam' : _regularization_penalty('regParam'),
        'elasticNetParam': _elastic_net_ratio('elasticNetParam')
    },
    {
        'type': 'decision_tree_regression',
        'maxDepth' : _trees_max_depth('decision_tree.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('decision_tree.minInstancesPerNode')
    },
    {
        'type': 'randomforest_regression',
        'subsamplingRate': _grad_boosting_subsample('randomforest.subsamplingRate'),
        'numTrees' :  _trees_n_estimators('randomforest.numTrees'),
        'maxDepth': _trees_max_depth('randomforest.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('randomforest.minInstancesPerNode')
    },
    {
        'type': 'grad_boosting_regression',
        'stepSize': _grad_boosting_learning_rate('grad_boosting.stepSize'),
        'subsamplingRate': _grad_boosting_subsample('grad_boosting.subsamplingRate'),
        'maxDepth': _trees_max_depth('grad_boosting.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('grad_boosting.minInstancesPerNode')
    }
])