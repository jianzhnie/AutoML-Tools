# Auto-sklearn

Auto-sklearn` 提供了开箱即用的监督型自动机器学习。从名字可以看出，Auto-Sklearn主要基于sklearn机器学习库，使用方法也与之类似，这让熟悉sklearn的开发者很容易切换到Auto-Sklearn。在模型方面，除了sklearn提供的机器学习模型，还加入了xgboost算法支持；在框架整体调优方面，使用了贝叶斯优化。

该库由 Matthias Feurer 等人提出，技术细节请查阅论文《Efficient and Robust Machine Learning》。Feurer 在这篇论文中写道：我们提出了一个新的、基于 scikit-learn 的鲁棒 AutoML 系统，其中使用 15 个分类器、14 种特征预处理方法和 4 种数据预处理方法，生成了一个具有 110 个超参数的结构化假设空间。

auto-sklearn 可能最适合刚接触 AutoML 的用户。除了发现数据集的数据准备和模型选择之外，该库还可以从在类似数据集上表现良好的模型中学习。表现最好的模型聚集在一个集合中。

![img](http://image.techweb.com.cn/upload/roll/2020/09/27/202009279365_9585.png)

​                                   图源：《Efficient and Robust Automated Machine Learning》

在高效实现方面，auto-sklearn 需要的用户交互最少。使用 pip install auto-sklearn 即可安装库。

该库可以使用的两个主要类是 AutoSklearnClassifier 和 AutoSklearnRegressor，它们分别用来做分类和回归任务。两者具有相同的用户指定参数，其中最重要的是时间约束和集合大小。



教程：https://machinelearningmastery.com/auto-sklearn-for-automated-machine-learning-in-python/

https://machinelearningmastery.com/what-is-bayesian-optimization/

## Auto-sklearn的整体框架

- 16 classifiers（可以被指定或者筛选，include_estimators=[“random_forest”, ]）
  
- adaboost, bernoulli_nb, decision_tree, extra_trees, gaussian_nb, gradient_boosting, k_nearest_neighbors, lda, liblinear_svc, libsvm_svc, multinomial_nb, passive_aggressive, qda, random_forest, sgd, xgradient_boosting
  
- 13 regressors（可以被指定或者筛选，exclude_estimators=None）

  - adaboost, ard_regression, decision_tree, extra_trees, gaussian_process, gradient_boosting, k_nearest_neighbors, liblinear_svr, libsvm_svr, random_forest, ridge_regression, sgd, xgradient_boosting

- 18 feature preprocessing methods（这些过程可以被手动关闭全部或者部分，include_preprocessors=[“no_preprocessing”, ]）

  - densifier, extra_trees_preproc_for_classification, extra_trees_preproc_for_regression, fast_ica,feature_agglomeration, kernel_pca, kitchen_sinks, liblinear_svc_preprocessor, no_preprocessing, nystroem_sampler, pca, polynomial, random_trees_embedding, select_percentile, select_percentile_classification, select_percentile_regression, select_rates, truncatedSVD

- 5 data preprocessing methods（这些过程不能被手动关闭）

  - balancing, imputation, one_hot_encoding, rescaling, variance_threshold（看到这里已经有点惊喜了！点进去有不少内容）

  - **more than 110 hyperparameters**
    其中参数include_estimators,要搜索的方法,exclude_estimators:为不搜索的方法.与参数include_estimators不兼容
    而include_preprocessors,可以参考手册中的内容

  auto-sklearn是基于sklearn库，因此会有惊艳强大的模型库和数据/特征预处理库，专业出身的设定。

##  Auto-sklearn 如何实现自动超参数调参

概念解释

- SMBO: Sequential Model-based Bayesian/Global Optimization，调超参的大多数方法基于SMBO
- SMAC: Sequential Model-based Algorithm Configuration，机器学习记录经验值的配置空间
- TPE: Tree-structured Parzen Estimator

超参数调参方法：

1. **Grid Search** 网格搜索/穷举搜索
   在高维空间不实用。

2. **Random Search** 随机搜索
   很多超参是通过并行选择的，它们之间是相互独立的。一些超参会产生良好的性能，另一些不会。

3. **Heuristic Tuning** 手动调参
   经验法，耗时长。（不知道经验法的英文是否可以这样表示）

4. Automatic Hyperparameter Tuning

   - Bayesian Optimization

   - 能利用先验知识高效地调节超参数
   - 通过减少计算任务而加速寻找最优参数的进程
   - 不依赖人为猜测所需的样本量为多少，优化技术基于随机性，概率分布
   - 在目标函数未知且计算复杂度高的情况下极其强大
   - 通常适用于连续值的超参，例如 learning rate, regularization coefficient

   - SMAC
   - TPE

