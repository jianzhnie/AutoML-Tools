## [TPOT](https://github.com/EpistasisLab/tpot)

TPOT是一个 Python 编写的软件包，利用遗传算法行特征选择和算法模型选择，仅需几行代码，就能生成完整的机器学习代码。

TPOT 是另一种基于 Python 的自动机器学习开发工具，该工具更关注数据准备、建模算法和模型超参数。它通过一种基于进化树的结，即自动设计和优化机器学习 pipelie 的树表示工作流优化(Tree-based Pipeline Optimization Tool, TPOT)，从而实现特征选择、预处理和构建的自动化。

![img](http://image.techweb.com.cn/upload/roll/2020/09/27/202009279899_8178.png)

图源：《Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science》 。

程序或 pipeline 用树表示。遗传编程(Genetic Program, GP)选择并演化某些程序，以最大化每个自动化机器学习管道的最终结果。

正如 Pedro Domingos 所说，「数据量大的愚蠢算法胜过数据有限的聪明算法」。事实就是这样：TPOT 可以生成复杂的数据预处理 pipeline。

![img](http://image.techweb.com.cn/upload/roll/2020/09/27/20200927210_9948.png)

潜在的 pipelie(图源：TPOT 文档)

TPOT pipeline 优化器可能需要几个小时才能产生很好的结果，就像很多 AutoML 算法一样(除非数据集很小)。用户可以在 Kaggle commits 或 Google Colab 中运行这些耗时的程序。

也许 TPOT 最好的特性是它将模型导出为 Python 代码文件，后续可以使用它。具体文档和教程示例参见以下两个链接：

TPOT 文档地址：https://epistasislab.github.io/tpot/。

 TPOT 的教程示例地址：https://epistasislab.github.io/tpot/examples/

