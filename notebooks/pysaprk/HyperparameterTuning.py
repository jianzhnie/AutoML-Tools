#!/usr/bin/env python
# coding: utf-8

# ## Hyperparameter Tuning with MMLSpark¶
# We can do distributed randomized grid search hyperparameter tuning with MMLSpark.
# 
# First, we import the packages

# ## Dataset Review

# The Adult dataset we are going to use is publicly available at the UCI Machine Learning Repository. This data derives from census data, and consists of information about 48842 individuals and their annual income. We will use this information to predict if an individual earns <=50K or >50k a year. The dataset is rather clean, and consists of both numeric and categorical variables.
# 
# Attribute Information:
# 
# - age: continuous
# - workclass: Private,Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
# - fnlwgt: continuous
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc...
# - education-num: continuous
# - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent...
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners...
# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# - sex: Female, Male
# - capital-gain: continuous
# - capital-loss: continuous
# - hours-per-week: continuous
# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany...
# - Target/Label: - <=50K, >50K

# In[1]:


from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.sql import SparkSession
import pyspark
spark = SparkSession.builder.appName("MyApp") \
    .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:0.18.1") \
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
    .getOrCreate()
import mmlspark


# Now let's read the data and split it to tuning and test sets:

# In[2]:


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
cols = dataset.columns
dataset.limit(10).toPandas()


# ## Preprocess Data
# 
# Since we are going to try algorithms like Logistic Regression, we will have to convert the categorical variables in the dataset into numeric variables. There are 2 ways we can do this.
# 
# - Category Indexing
# 
# This is basically assigning a numeric value to each category from {0, 1, 2, ...numCategories-1}. This introduces an implicit ordering among your categories, and is more suitable for ordinal variables (eg: Poor: 0, Average: 1, Good: 2)
# 
# - One-Hot Encoding
# 
# This converts categories into binary vectors with at most one nonzero value (eg: (Blue: [1, 0]), (Green: [0, 1]), (Red: [0, 0]))
# 
# In this dataset, we have ordinal variables like education (Preschool - Doctorate), and also nominal variables like relationship (Wife, Husband, Own-child, etc). For simplicity's sake, we will use One-Hot Encoding to convert all categorical variables into binary vectors. It is possible here to improve prediction accuracy by converting each categorical column with an appropriate method.
# 
# Here, we will use a combination of StringIndexer and OneHotEncoderEstimator to convert the categorical variables. The OneHotEncoderEstimator will return a SparseVector. Note: OneHotEncoderEstimator is renamed as OneHotEncoder in Spark 3.0.
# 
# Since we will have more than 1 stage of feature transformations, we use a Pipeline to tie the stages together. This simplifies our code.

# In[3]:


import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

from distutils.version import LooseVersion

categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]


# The above code basically indexes each categorical column using the StringIndexer, and then converts the indexed categories into one-hot encoded variables. The resulting output has the binary vectors appended to the end of each row.
# 
# We use the StringIndexer again to encode our labels to label indices.

# In[4]:


# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]


# Use a VectorAssembler to combine all the feature columns into a single vector column. This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.

# In[5]:


# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# Run the stages as a Pipeline. This puts the data through all of the feature transformations we described in a single call.

# In[6]:


partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)
preppedDataDF.limit(10).toPandas()


# In[7]:


# Keep relevant columns
selectedcols = ["label", "features"]
dataset = preppedDataDF.select(selectedcols)
dataset.limit(10).toPandas()


# In[8]:


### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())


# In[9]:


from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
# Train model with Training Data
lrModel = lr.fit(trainingData)


# In[10]:


# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)


# In[11]:


# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well. For example's sake we will choose age & occupation
selected = predictions.select("label", "prediction", "probability")




# In[12]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)


# ## Hyperparameter Tuning with MMLSpark

# In[13]:


from mmlspark.automl import TuneHyperparameters
from mmlspark.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
logReg = LogisticRegression()
randForest = RandomForestClassifier()
gbt = GBTClassifier()
smlmodels = [logReg, randForest, gbt]
mmlmodels = [TrainClassifier(model=model, featuresCol="features",labelCol="label") for model in smlmodels]


# We can specify the hyperparameters using the HyperparamBuilder. We can add either DiscreteHyperParam or RangeHyperParam hyperparameters. TuneHyperparameters will randomly choose values from a uniform distribution.

# In[14]:


from mmlspark.automl import *

paramBuilder =   HyperparamBuilder().addHyperparam(logReg, logReg.regParam, RangeHyperParam(0.1, 0.3))     .addHyperparam(randForest, randForest.numTrees, DiscreteHyperParam([5,10]))     .addHyperparam(randForest, randForest.maxDepth, DiscreteHyperParam([3,5]))     .addHyperparam(gbt, gbt.maxBins, RangeHyperParam(8,16))     .addHyperparam(gbt, gbt.maxDepth, DiscreteHyperParam([3,5]))
searchSpace = paramBuilder.build()
# The search space is a list of params to tuples of estimator and hyperparam
print(searchSpace)
randomSpace = RandomSpace(searchSpace)


# Next, run TuneHyperparameters to get the best model.

# In[15]:


bestModel = TuneHyperparameters(
              evaluationMetric="accuracy", models=mmlmodels, numFolds=2,
              numRuns=len(mmlmodels) * 2, parallelism=1,
              paramSpace=randomSpace.space(), seed=0).fit(trainingData)


# In[ ]:





# In[ ]:




