# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 106 - Quantile Regression with LightGBM
# 
# We will demonstrate how to use the LightGBM quantile regressor with
# TrainRegressor and ComputeModelStatistics on the Triazines dataset.
# 
# 
# This sample demonstrates how to use the following APIs:
# - [`TrainRegressor`
#   ](http://mmlspark.azureedge.net/docs/pyspark/TrainRegressor.html)
# - [`LightGBMRegressor`
#   ](http://mmlspark.azureedge.net/docs/pyspark/LightGBMRegressor.html)
# - [`ComputeModelStatistics`
#   ](http://mmlspark.azureedge.net/docs/pyspark/ComputeModelStatistics.html)

# %%
triazines = spark.read.format("libsvm")    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/triazines.scale.svmlight")


# %%
# print some basic info
print("records read: " + str(triazines.count()))
print("Schema: ")
triazines.printSchema()
triazines.limit(10).toPandas()

# %% [markdown]
# Split the dataset into train and test

# %%
train, test = triazines.randomSplit([0.85, 0.15], seed=1)

# %% [markdown]
# Train the quantile regressor on the training data.

# %%
from mmlspark.lightgbm import LightGBMRegressor
model = LightGBMRegressor(objective='quantile',
                          alpha=0.2,
                          learningRate=0.3,
                          numLeaves=31).fit(train)

# %% [markdown]
# We can save and load LightGBM to a file using the LightGBM native representation

# %%
from mmlspark.lightgbm import LightGBMRegressionModel
model.saveNativeModel("mymodel")
model = LightGBMRegressionModel.loadNativeModelFromFile("mymodel")

# %% [markdown]
# View the feature importances of the trained model.

# %%
print(model.getFeatureImportances())

# %% [markdown]
# Score the regressor on the test data.

# %%
scoredData = model.transform(test)
scoredData.limit(10).toPandas()

# %% [markdown]
# Compute metrics using ComputeModelStatistics

# %%
from mmlspark.train import ComputeModelStatistics
metrics = ComputeModelStatistics(evaluationMetric='regression',
                                 labelCol='label',
                                 scoresCol='prediction') \
            .transform(scoredData)
metrics.toPandas()


