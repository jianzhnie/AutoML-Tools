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
from pyspark.ml.feature import StringIndexer, VectorAssembler,OneHotEncoder
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, 
                GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, Evaluator
from search_space import cls_space
from estimater import ClsObjective
from functools import partial



experiment_name = "adult_income_classification"
mlflow.set_experiment(experiment_name)

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Binary Classification") \
        .getOrCreate()

    schema = StructType([
        StructField("age", DoubleType(), False),
        StructField("workclass", StringType(), False),
        StructField("fnlwgt", DoubleType(), False),
        StructField("education", StringType(), False),
        StructField("education_num", DoubleType(), False),
        StructField("marital_status", StringType(), False),
        StructField("occupation", StringType(), False),
        StructField("relationship", StringType(), False),
        StructField("race", StringType(), False),
        StructField("sex", StringType(), False),
        StructField("capital_gain", DoubleType(), False),
        StructField("capital_loss", DoubleType(), False),
        StructField("hours_per_week", DoubleType(), False),
        StructField("native_country", StringType(), False),
        StructField("income", StringType(), False)
    ])


    dataset = spark.read.format("csv").schema(schema).load("/home/robin/datatsets/adult/adult.data")

    categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    stages = [] # stages in our Pipeline
    for categoricalCol in categoricalColumns:
        # Category Indexing with StringIndexer
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
        # Use OneHotEncoder to convert categorical variables into binary SparseVectors
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        # Add stages.  These are not run here, but will run all at once later on.
        stages += [stringIndexer, encoder]


    label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
    stages += [label_stringIdx]

    numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    partialPipeline = Pipeline().setStages(stages)
    pipelineModel = partialPipeline.fit(dataset)
    preparedDataDF = pipelineModel.transform(dataset)

    # Keep relevant columns
    selectedcols = ["label", "features"]
    dataset = preparedDataDF.select(selectedcols)

    ### Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

    # Evaluate model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    fmin_objective = partial(ClsObjective, trainData=trainingData, testData=testData, evaluator=evaluator)

    # Trials object to track progress
    bayes_trials = Trials()
    # Select a search algorithm for Hyperopt to use.
    algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

    # We can run Hyperopt locally (only on the driver machine)
    # by calling `fmin` without an explicit `trials` argument.
    with mlflow.start_run(nested=True):
        mlflow.set_tag("mlflow.runName", "hyperopt run")
        best_hyperparameters = fmin(fn=fmin_objective,
                                    space=cls_space,
                                    algo=algo,
                                    max_evals=100,
                                    trials=bayes_trials,
                                    max_queue_len=10,
                                    timeout=6000)
        mlflow.set_tag("best params", str(best_hyperparameters))
    print(bayes_trials)
    print(best_hyperparameters)
    mlflow.end_run()