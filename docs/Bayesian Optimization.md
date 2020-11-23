# Auto Machine Learning笔记 - Bayesian Optimization

优化器是机器学习中很重要的一个环节。当确定损失函数时，你需要一个优化器使损失函数的参数能够快速有效求解成功。优化器很大程度影响计算效率。越来越多的超参数调整是通过自动化方式完成，使用明智的搜索在更短的时间内找到最佳超参组合，无需在初始设置之外进行手动操作。

**贝叶斯优化**（Bayesian Optimization）是基于模型的超参数优化，已应用于机器学习超参数调整，结果表明该方法可以在测试集上实现更好的性能，同时比随机搜索需要更少的迭代。此外，现在有许多Python库可以为任何机器学习模型简化实现贝叶斯超参数调整。
[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/001.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/001.png?raw=true)

## 1. 超参数是什么？

- 在模型开始学习过程之前人为设置值的参数，而不是（像bias、weights）通过训练可得到的参数数据。
- 这些参数定义关于模型更高层次的概念（模型复杂性、学习能力等）。
- 比如说随机梯度下降算法中的学习速率/learning rate，出于计算复杂度和算法效率等，我们并不能从数据中直接学习一个比较不错的学习速度。但学习速率却又是十分重要的，较大的学习速率不易令模型收敛到较合适的较小值解，而较小的学习速率却又常常令模型的训练速度大大降低。对于像学习速率这样的超参数，我们通常需要在训练模型之前设定。因此，对于超参数众多的复杂模型，调超参技能显得很重要。

## 2. 常用的调超参方法有哪些？

1. Grid Search

   网格搜索/穷举搜索

   - 搜索整个超参数空间，在高维空间容易遇到维度灾难，不实用。
   - 网格搜索是一种昂贵的方法。假设我们有n个超参数，每个超参数有两个值，那么配置总数就是2的N次方。因此，仅在少量配置上进行网格搜索是可行的。
   - 网格搜索可以并行化，使得网格搜索在足够的计算能力下更加可行。
   - 每次trial之间是相互独立的，不能利用先验知识选择下一组超参数。

2. Random Search

   随机搜索

   - 稀疏的简单抽样，试验之间是相互独立的，不能利用先验知识选择下一组超参数。
   - 超参通过并行选择，但试验次数要少得多，而性能却相当。一些超参可能会产生良好的性能，另一些不会。

3. Heuristic Tuning

    

   手动调参

   - 经验法，耗时长。

4. Automatic Hyperparameter Tuning

    

   自动超参数调优

   - 自动超参数调整形成了关于超参数设置和模型性能之间关系的知识，能利用先验知识选择下一组超参数。
   - 首先在多个配置中收集性能，然后进行一些推断并确定接下来要尝试的配置。目的是在找到最佳状态时尽量减少试验次数。
   - 这个过程本质上是顺序的，不容易并行化。
   - 调整超参数的大多数方法都属于基于顺序模型的全局优化（SMBO）。这些方法使用代理函数来逼近真正的黑盒函数。SMBO的内部循环是对该替代品的优化，或者对代理进行某种转换。最大化此代理的配置将是下一个应该尝试的配置。SMBO算法在优化替代品的标准以及他们根据观察历史对替代品进行建模的方式上有所不同。最近在文献中提出了几种用于超参数的SMBO方法：
     - **Bayesian Optimization**
       使用*高斯过程*对代理进行建模，通常优化 Expected Improvement(EI)，这是新试验将在当前最佳观察上改进的预期概率。高斯过程是函数的分布。来自高斯过程的样本是整个函数。训练高斯过程涉及将此分布拟合到给定数据，以便生成接近观察数据的函数。使用高斯过程，可以计算搜索空间中任何点的EI。接下来将尝试给出最高的EI。贝叶斯优化通常为连续超参数（例如learning rate, regularization coefficient…）提供 non-trivial/off-the-grid 值，并且在一些好的数据集上击败人类表现。Spearmint是一个众所周知的贝叶斯优化实现。
     - **SMAC**
       使用*随机森林*对目标函数进行建模，从随机森林认为最优的区域（高EI）中抽取下一个点。
     - **TPE**
       是SMAC的改进版本，其中两个分离的模型用于模拟后验。众所周知的TPE实现是hyperopt。

## 3. 概念解释

- 高斯过程
  [![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/010.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/010.png?raw=true)
  上图里那么多线就是高斯过程的体现。
  要使用贝叶斯优化，需要一种能灵活地在目标函数上建立分布的方法。这比在实数上建立分布更棘手，因为我们需要一个这样的分布来表示我们对每个x的f(x)的信念。如果x包含连续的超参数，那么我们必须为f(x)建模无限多的x，即构造它的分布。对于这个问题，**高斯过程**是一种优雅的方法。实际上，高斯过程生成多维高斯分布，并且存在足够灵活以模拟任何目标函数的模样。
  [![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/013.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/013.png?raw=true)

- Prior Function 先验函数
  基于概率分布，用于描述目标函数的分布，拟合目标函数曲线。不同的分布，PF不同，效果是不一样的。

- Acquisition Function 收获函数 = max(mean + var)

  用于从候选集中选择一个新的点。贝叶斯优化的效果与AF的设计有较大的关系，由于此类function可能陷入局部最优解，因此在选点时，需考虑不能过早进入局部最优。AF计算EI，用来选择下一个采样点。

  - mean均值大：
    多去采样这些点会帮助我们更好的了解这个函数形态。
  - var方差大：
    表示我们对该点的了解甚少。

- 采样点

  每一个采样点就是原理解析里的黑点。每个采样点是基于前面n个点的多变量高斯分布的假设以及最大化AF而得到的，现目前为止认为的y的最大值最可能出现的位置。

  - 一开始，采样数据少，算法会采标准差大的点。采样一定数目后，标准差的值会下降很多，此时采样点的选择就更多的受到均值的影响，采样点就更大概率的出现在真正最大值附近。

## 4. Bayesian Optimizer 原理解析

贝叶斯优化基于高斯过程。
[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/010.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/010.png?raw=true)
[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/003.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/003.png?raw=true)

- 上图2个evaluations黑点，是两次评估后显示替代模型的初始值估计，会影响下一个点的选择，穿过这两个点的曲线可以画出非常多条，如上上图
- 红色虚线曲线是实际真正的目标函数
- 黑色实线曲线是代理模型的目标函数的均值
- 灰色区域是代理模型的目标函数的方差
- 只有两个点，拟合的效果稍差，根据下方的紫色的EI曲线，最左侧的最大值EI为下一个点

[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/004.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/004.png?raw=true)

- 3个evaluations黑点
- 灰色区域是代理模型的目标函数的方差，黑点越多，灰色区域面积越小，误差越小
- 根据下方的紫色的EI曲线，左侧的最大值EI为第四个拟合点

[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/005.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/005.png?raw=true)

- 4个evaluations黑点
- 黑点越多，灰色区域面积越小，误差越小，代理模型越接近真实模型的目标函数
- 根据下方的紫色的EI曲线，最大值EI为第五个拟合点，同理类推…

[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/006.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/006.png?raw=true)
[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/007.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/007.png?raw=true)
[![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/008.png?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/008.png?raw=true)

- 8个黑点
- 黑色代理曲线已经十分接近红色真实目标函数，灰色区域也越来越小，拟合效果不错。

## 5. Bayesian Optimizer 基本思想

一句话总结：建立目标函数的概率模型，并用它来选择最有希望的超参数来评估真实的目标函数。
基本思想是：利用先验知识逼近未知目标函数的后验分布从而调节超参。花一点时间选择下一个超参数，以减少对目标函数的调用。

1. 建立代理模型的目标函数（Prior Function/先验函数）
2. 找到在代理上表现最佳的超参数（利用EI值，根据Acquisition Function得出EI）
3. 将这些超参数应用于真正的目标函数
4. 更新包含新结果的代理模型
5. 重复步骤2-4，直到达到最大迭代次数或时间

基于顺序模型的优化方法（SMBO）是贝叶斯优化的形式化。顺序是指一个接一个地运行试验，每次通过应用贝叶斯推理和更新概率模型（代理）来尝试更好的超参数。

## 6. Bayesian Optimizer 在python中的包

Python中有几个贝叶斯优化库，它们在目标函数的代理算法上有所不同。

- Spearmint（高斯过程代理）
- SMAC（随机森林回归）
- Hyperopt（Tree Parzen Estimator-TPE）

## 7. Bayesian Optimizer 优点

- 能利用先验知识高效地调节超参数，每个试验不独立，前一个推动下一个选择

- 通过减少计算任务而加速寻找最优参数的进程

- 不依赖人为猜测所需的样本量为多少，优化技术基于随机性，概率分布

- 在目标函数未知且计算复杂度高的情况下极其强大

- 通常适用于连续值的超参，例如 learning rate, regularization coefficient

- 在测试集表现优秀于手工调参结果，泛化性/鲁棒性好

- 不易陷入局部最优：

  EI的计算根据 Acquisition Function收获函数计算所得：

  - **探索**和**开发**是解释现象和优化算法的常用术语。
  - 对于贝叶斯优化，一旦找到局部最优解，会在这个区域不断采样，很容易陷入局部最优。为了减轻这个，贝叶斯优化会使用“ 收获函数Acquisition Function ” 来平衡“探索”和“开发”，下一个点的选择要在这两者之间存在权衡。
  - 下一个选择点（x）应该具有高均值（开发）和高方差（探索）。
    [![img](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian%20optimizer/013.jpeg?raw=true)](https://github.com/YZHANG1270/Markdown_pic/blob/master/2018/07/bayesian optimizer/013.jpeg?raw=true)

## 8. 其他优秀文章与论文链接🔗

- [贝叶斯优化: 一种更好的超参数调优方式](https://zhuanlan.zhihu.com/p/29779000)
- [哈佛教材：A Tutorial on Bayesian Optimization for Machine Learning](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf)
- [Shallow Understanding on Bayesian Optimization](https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083)
- [谷歌cloudml也在用贝叶斯优化](https://cloud.google.com/blog/products/gcp/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization)
- [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1012.2599.pdf)
- [Practical Bayesian Optimization of Machine Learning Algorithms](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
- [Automated Machine Learning Hyperparameter Tuning in Python](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a)
- [A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
- [Introduction to Bayesian Optimization](http://gpss.cc/gpmc17/slides/LancasterMasterclass_1.pdf)