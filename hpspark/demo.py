from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Binary Classification") \
    .getOrCreate()

from components_ import any_classifier


if __name__ == "__main__":
    clf = any_classifier("cls")
    print(clf)
    print(type(clf))