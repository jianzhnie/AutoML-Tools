# -*- coding: utf-8 -*-
from __future__ import print_function
import mlflow
from hyperopt import Trials, STATUS_OK
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, 
                GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.regression import (LinearRegression, DecisionTreeRegressor, 
                GBTRegressor, RandomForestRegressor)


def ClsObjective(params, trainData, testData, evaluator, labelCol="label", featuresCol="features"):
    """
    This is our main training function which we pass to Hyperopt.
    It takes in hyperparameter settings, fits a model based on those settings,
    evaluates the model, and returns the loss.

    :param params: map specifying the hyperparameter settings to test
    :return: loss for the fitted model
    """
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model_type = params['type']
        del params['type']
        if model_type == 'decision_tree':
            estimator = DecisionTreeClassifier(**params, labelCol=labelCol, featuresCol=featuresCol)
        elif model_type == 'randomforest':
            estimator = RandomForestClassifier(**params, labelCol=labelCol, featuresCol=featuresCol)
        elif model_type == 'grad_boosting':
            estimator = GBTClassifier(**params, labelCol=labelCol, featuresCol=featuresCol)
        else:
            raise Exception("Not included this method") 

        Model = estimator.fit(trainData)
        predictions = Model.transform(testData)
        score = evaluator.evaluate(predictions)
        mlflow.log_metric("score", score)
    return {'loss': -score, 'status': STATUS_OK}


def RegObjective(params, trainData, testData, evaluator, labelCol="label", featuresCol="features"):
    """
    This is our main training function which we pass to Hyperopt.
    It takes in hyperparameter settings, fits a model based on those settings,
    evaluates the model, and returns the loss.

    :param params: map specifying the hyperparameter settings to test
    :return: loss for the fitted model
    """
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model_type = params['type']
        del params['type']
        if model_type == 'linear_regression':
            estimator = LinearRegression(**params, labelCol=labelCol, featuresCol=featuresCol)
        elif model_type == 'decision_tree_regression':
            estimator = DecisionTreeRegressor(**params, labelCol=labelCol, featuresCol=featuresCol)
        elif model_type == 'randomforest_regression':
            estimator = RandomForestRegressor(**params, labelCol=labelCol, featuresCol=featuresCol)
        elif model_type == 'grad_boosting_regression':
            estimator = GBTRegressor(**params, labelCol=labelCol, featuresCol=featuresCol)
        else:
            raise Exception("Not included this method") 

        Model = estimator.fit(trainData)
        predictions = Model.transform(testData)
        score = evaluator.evaluate(predictions)
        mlflow.log_metric("score", score)
    return {'loss': score, 'status': STATUS_OK}