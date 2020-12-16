# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import hyperopt
import mlflow
from hyperopt import fmin, hp, tpe
from hyperopt import Trials, STATUS_OK

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.ml.feature import StringIndexer, VectorAssembler,OneHotEncoder, VectorIndexer
from pyspark.ml.regression import (LinearRegression, DecisionTreeRegressor, 
                GBTRegressor, RandomForestRegressor)
from pyspark.ml.evaluation import RegressionEvaluator
from search_space import reg_space
from estimater import RegObjective
from functools import partial

"""
root
 |-- season: integer (nullable = true)
 |-- yr: integer (nullable = true)
 |-- mnth: integer (nullable = true)
 |-- hr: integer (nullable = true)
 |-- holiday: integer (nullable = true)
 |-- weekday: integer (nullable = true)
 |-- workingday: integer (nullable = true)
 |-- weathersit: integer (nullable = true)
 |-- temp: double (nullable = true)
 |-- atemp: double (nullable = true)
 |-- hum: double (nullable = true)
 |-- windspeed: double (nullable = true)
 |-- cnt: integer (nullable = true)

instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt

"""


experiment_name = "bikeSharing_Regression"
mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Regression") \
        .getOrCreate()

    df = spark.read.csv("/home/robin/datatsets/bikeSharing/hour.csv", header="true", inferSchema="true")
    """
        root
    |-- instant: integer (nullable = true)
    |-- dteday: string (nullable = true)
    |-- season: integer (nullable = true)
    |-- yr: integer (nullable = true)
    |-- mnth: integer (nullable = true)
    |-- hr: integer (nullable = true)
    |-- holiday: integer (nullable = true)
    |-- weekday: integer (nullable = true)
    |-- workingday: integer (nullable = true)
    |-- weathersit: integer (nullable = true)
    |-- temp: double (nullable = true)
    |-- atemp: double (nullable = true)
    |-- hum: double (nullable = true)
    |-- windspeed: double (nullable = true)
    |-- casual: integer (nullable = true)
    |-- registered: integer (nullable = true)
    |-- cnt: integer (nullable = true)
    """
    df.printSchema()
    # Preprocess data
    df = df.drop("instant").drop("dteday").drop("casual").drop("registered")

    # Remove the target column from the input feature set.
    featuresCols = df.columns
    featuresCols.remove('cnt')

    # vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
    
    # vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
    vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

    #label_vectorAssembler = VectorAssembler(inputCols=["cnt"], outputCol="label")
    
    pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer])
    pipelineModel = pipeline.fit(df)
    preparedDataDF = pipelineModel.transform(df)
    preparedDataDF.printSchema()

    # Keep relevant columns
    selectedcols = ["cnt", "features"]
    df = preparedDataDF.select(selectedcols)

    # Split the dataset randomly into 70% for training and 30% for testing. Passing a seed for deterministic behavior
    train, test = df.randomSplit([0.7, 0.3], seed = 0)
    print("There are %d training examples and %d test examples." % (train.count(), test.count()))

    # Evaluate model
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="cnt")
    fmin_objective = partial(RegObjective, trainData=train, testData=test, evaluator=evaluator, labelCol="cnt", featuresCol="features")

    # Trials object to track progress
    bayes_trials = Trials()
    # Select a search algorithm for Hyperopt to use.
    algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

    # We can run Hyperopt locally (only on the driver machine)
    # by calling `fmin` without an explicit `trials` argument.
    with mlflow.start_run(run_name='bikeSharing_Regression', nested=True):
        best_hyperparameters = fmin(fn=fmin_objective,
                                    space=reg_space,
                                    algo=algo,
                                    max_evals=100,
                                    trials=bayes_trials,
                                    max_queue_len=10,
                                    timeout=6000)
        mlflow.set_tag("best params", str(best_hyperparameters))
    print(bayes_trials)
    print(best_hyperparameters)
    mlflow.end_run()