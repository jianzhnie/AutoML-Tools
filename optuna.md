# Optuna: 一个超参数优化框架

*Optuna* 是一个特别为机器学习设计的自动超参数优化软件框架。它具有命令式的，*define-by-run* 风格的 API。由于这种 API 的存在，用 Optuna 编写的代码模块化程度很高，Optuna 的用户因此也可以动态地构造超参数的搜索空间。

## 主要特点

Optuna 有如下现代化的功能：

- [轻量级、多功能和跨平台架构](https://tigeraus.gitee.io/doc-optuna-chinese-build/tutorial/first.html)
- [并行的分布式优化](https://tigeraus.gitee.io/doc-optuna-chinese-build/tutorial/distributed.html)
- [对不理想实验 (trial) 的剪枝 (pruning)](https://tigeraus.gitee.io/doc-optuna-chinese-build/tutorial/pruning.html)

### **使用数据库存储搜索过程**

Optuna 后端使用 SQL 存储实验记录，这种持久化的存储可以很方便我们访问研究历史记录。

#### New Study

我们可以通过调用create_study（）函数来创建持久性的 study，如下所示。 SQLite文件example.db使用新的研究记录自动初始化。

```python
import optuna
study_name = 'example-study'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')
```

To run a study, call [`optimize()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize) method passing an objective function.

```python
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

study.optimize(objective, n_trials=3)
```

#### Resume Study

To resume a study, instantiate a [`Study`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study) object passing the study name `example-study` and the DB URL `sqlite:///example.db`.

```python
study = optuna.create_study(study_name='example-study', storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=3)
```

#### Experimental History

We can access histories of studies and trials via the [`Study`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study) class. For example, we can get all trials of `example-study` as:

```python
import optuna
study = optuna.create_study(study_name='example-study', storage='sqlite:///example.db', load_if_exists=True)
df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
```

The method [`trials_dataframe()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.trials_dataframe) returns a pandas dataframe like:

```python
print(df)
```

Out:

```python
   number       value  params_x     state
0       0   25.301959 -3.030105  COMPLETE
1       1    1.406223  0.814157  COMPLETE
2       2   44.010366 -4.634031  COMPLETE
3       3   55.872181  9.474770  COMPLETE
4       4  113.039223 -8.631991  COMPLETE
5       5   57.319570  9.570969  COMPLETE
```

### Distributed Optimization

First, create a shared study using `optuna create-study` command (or using [`optuna.create_study()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html#optuna.create_study) in a Python script).

```python
$ optuna create-study --study-name "distributed-example" --storage "sqlite:///example.db"          

[I 2020-11-09 23:46:20,540] A new study created in RDB with name: di6:20,540] A new study created in RDB with name: distributed-example
```

Then, write an optimization script. Let’s assume that `foo.py` contains the following code.

```python
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

if __name__ == '__main__':
    study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=100)
```

Finally, run the shared study from multiple processes. For example, run `Process 1` in a terminal, and do `Process 2` in another one. They get parameter suggestions based on shared trials’ history.

Process 1:

```python
$ python foo.py
[I 2020-07-21 13:45:02,973] Trial 0 finished with value: 45.35553104173011 and parameters: {'x': 8.73465151598285}. Best is trial 0 with value: 45.35553104173011.
[I 2020-07-21 13:45:04,013] Trial 2 finished with value: 4.6002397305938905 and parameters: {'x': 4.144816945707463}. Best is trial 1 with value: 0.028194513284051464.
...
```

Process 2 (the same command as process 1):

```python
$ python foo.py
[I 2020-07-21 13:45:03,748] Trial 1 finished with value: 0.028194513284051464 and parameters: {'x': 1.8320877810162361}. Best is trial 1 with value: 0.028194513284051464.
[I 2020-07-21 13:45:05,783] Trial 3 finished with value: 24.45966755098074 and parameters: {'x': 6.945671597566982}. Best is trial 1 with value: 0.028194513284051464.
...
```

### Pruning Unpromising Trials

This feature automatically stops unpromising trials at the early stages of the training (a.k.a., automated early-stopping). Optuna provides interfaces to concisely implement the pruning mechanism in iterative training algorithms.

#### Activating Pruners

To turn on the pruning feature, you need to call [`report()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report) and [`should_prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune) after each step of the iterative training. [`report()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report) periodically monitors the intermediate objective values. [`should_prune()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune) decides termination of the trial that does not meet a predefined condition.

```python
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import optuna


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = \
        sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)
```

Set up the median stopping rule as the pruning condition.

```python
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
```

Executing the script above:

```python
$ python prune.py
[I 2020-06-12 16:54:23,876] Trial 0 finished with value: 0.3157894736842105 and parameters: {'alpha': 0.00181467547181131}. Best is trial 0 with value: 0.3157894736842105.
[I 2020-06-12 16:54:23,981] Trial 1 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.015378744419287613}. Best is trial 1 with value: 0.07894736842105265.
[I 2020-06-12 16:54:24,083] Trial 2 finished with value: 0.21052631578947367 and parameters: {'alpha': 0.04089428832878595}. Best is trial 1 with value: 0.07894736842105265.
[I 2020-06-12 16:54:24,185] Trial 3 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.004018735937374473}. Best is trial 3 with value: 0.052631578947368474.
[I 2020-06-12 16:54:24,303] Trial 4 finished with value: 0.07894736842105265 and parameters: {'alpha': 2.805688697062864e-05}. Best is trial 3 with value: 0.052631578947368474.
[I 2020-06-12 16:54:24,315] Trial 5 pruned.
[I 2020-06-12 16:54:24,355] Trial 6 pruned.
[I 2020-06-12 16:54:24,511] Trial 7 finished with value: 0.052631578947368474 and parameters: {'alpha': 2.243775785299103e-05}. Best is trial 3 with value: 0.052631578947368474.
[I 2020-06-12 16:54:24,625] Trial 8 finished with value: 0.1842105263157895 and parameters: {'alpha': 0.007021209286214553}. Best is trial 3 with value: 0.052631578947368474.
[I 2020-06-12 16:54:24,629] Trial 9 pruned.
...
```

`Trial 5 pruned.`, etc. in the log messages means several trials were stopped before they finished all of the iterations.

#### Integration Modules for Pruning

To implement pruning mechanism in much simpler forms, Optuna provides integration modules for the following libraries.

For the complete list of Optuna’s integration modules, see [`integration`](https://optuna.readthedocs.io/en/stable/reference/integration.html#module-optuna.integration).

For example, [`XGBoostPruningCallback`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.XGBoostPruningCallback.html#optuna.integration.XGBoostPruningCallback) introduces pruning without directly changing the logic of training iteration. (See also [example](https://github.com/optuna/optuna/blob/master/examples/pruning/xgboost_integration.py) for the entire script.)

```python
pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-error')
bst = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], callbacks=[pruning_callback])
```

### User-Defined Sampler

Thanks to user-defined samplers, you can:

- experiment your own sampling algorithms,
- implement task-specific algorithms to refine the optimization performance, or
- wrap other optimization libraries to integrate them into Optuna pipelines (e.g., [`SkoptSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.SkoptSampler.html#optuna.integration.SkoptSampler)).

This section describes the internal behavior of sampler classes and shows an example of implementing a user-defined sampler.

#### Overview of Sampler

A sampler has the responsibility to determine the parameter values to be evaluated in a trial. When a suggest API (e.g., [`suggest_uniform()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform)) is called inside an objective function, the corresponding distribution object (e.g., [`UniformDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.UniformDistribution.html#optuna.distributions.UniformDistribution)) is created internally. A sampler samples a parameter value from the distribution. The sampled value is returned to the caller of the suggest API and evaluated in the objective function.

To create a new sampler, you need to define a class that inherits [`BaseSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler). The base class has three abstract methods; [`infer_relative_search_space()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.infer_relative_search_space), [`sample_relative()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.sample_relative), and [`sample_independent()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.sample_independent).

As the method names imply, Optuna supports two types of sampling: one is **relative sampling** that can consider the correlation of the parameters in a trial, and the other is **independent sampling** that samples each parameter independently.

At the beginning of a trial, [`infer_relative_search_space()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.infer_relative_search_space) is called to provide the relative search space for the trial. Then, [`sample_relative()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.sample_relative) is invoked to sample relative parameters from the search space. During the execution of the objective function, [`sample_independent()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler.sample_independent) is used to sample parameters that don’t belong to the relative search space.

#### An Example: Implementing SimulatedAnnealingSampler

For example, the following code defines a sampler based on [Simulated Annealing (SA)](https://en.wikipedia.org/wiki/Simulated_annealing):

```python
import numpy as np
import optuna


class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.RandomState()
        self._temperature = temperature  # Current temperature.
        self._current_trial = None  # Current state.

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        #
        # An implementation of SA algorithm.
        #

        # Calculate transition probability.
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp((self._current_trial.value - prev_trial.value) / self._temperature)
        self._temperature *= 0.9  # Decrease temperature.

        # Transit the current state if the previous result is accepted.
        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial

        # Sample parameters from the neighborhood of the current point.
        #
        # The sampled parameters will be used during the next execution of
        # the objective function passed to the study.
        params = {}
        for param_name, param_distribution in search_space.items():
            if not isinstance(param_distribution, optuna.distributions.UniformDistribution):
                raise NotImplementedError('Only suggest_uniform() is supported')

            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    #
    # The rest is boilerplate code and unrelated to SA algorithm.
    #
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)
```

You can use `SimulatedAnnealingSampler` in the same way as built-in samplers as follows:

```python
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -5, 5)
    return x**2 + y

sampler = SimulatedAnnealingSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

In this optimization, the values of `x` and `y` parameters are sampled by using `SimulatedAnnealingSampler.sample_relative` method.



## 基本概念

简单介绍一下optuna里最重要的几个term。

### **在optuna里最重要的三个term**：

（1）Trial：目标函数的单次执行过程

（2）Study：基于目标函数的优化过程, 一个优化超参的session，由一系列的trials组成；

（3）Parameter：需要优化的超参；

在optuna里，study对象用来管理对超参的优化，optuna.create_study()返回一个study对象。

###  **study又有很多有用的 property**：

（1）study.best_params：搜出来的最优超参；

（2）study.best_value：最优超参下，objective函数返回的值 (如最高的Acc，最低的Error rate等)；

（3）study.best_trial：最优超参对应的trial，有一些时间、超参、trial编号等信息；

（4）study.optimize(objective, n_trials)：对objective函数里定义的超参进行搜索；

### **optuna支持很多种搜索方式：**

（1）trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])：表示从SGD和adam里选一个使用；

（2）trial.suggest_int('num_layers', 1, 3)：从1～3范围内的int里选；

（3）trial.suggest_uniform('dropout_rate', 0.0, 1.0)：从0～1内的uniform分布里选；

（4）trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)：从1e-5～1e-2的log uniform分布里选；

（5）trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)：从0～1且step为0.1的离散uniform分布里选；

请参考下面的示例代码。一个 *study* 的目的是通过多次 *trial* (例如 `n_trials=100` ) 来找出最佳的超参数值集（比如选择 `classifier` 还是 `svm_c`）。而 Optuna 旨在加速和自动化此类 *study* 优化过程。

```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_loguniform('svr_c', 1e-10, 1e10)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```

## PyTorch Example Code

扒一个官方给的PyTorch例子，[链接在这](https://link.zhihu.com/?target=https%3A//github.com/optuna/optuna/blob/master/examples/pytorch_simple.py)：

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
```

到目前为止是一些package的导入和参数的定义；

```python
def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader
```

这一段代码里有个很有趣的用法，定义了一个函数define_model，里面用定义了一个需要优化的变量n_layers，用来搜索网络的层数，所以其实optuna还可以用来做NAS之类的工作，return成nn.Module就可以搜起来了；

```python
def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)

            # Zeroing out gradient buffers.
            optimizer.zero_grad()
            # Performing a forward pass.
            output = model(data)
            # Computing negative Log Likelihood loss.
            loss = F.nll_loss(output, target)
            # Performing a backward pass.
            loss.backward()
            # Updating the weights.
            optimizer.step()

    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Limiting validation data.
            if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES
    return accuracy
```

这里是最重要的objective函数，首先定义了几个需要优化的parameter，「optimizer_name, lr和model里的n_layers和p」。剩下的就是一些常规的训练和测试代码，其中N_TRAIN_EXAMPLES和N_VAL_EXAMPLES是为了筛选出一小部分数据集用来搜索，毕竟用整个数据集来搜还是挺费劲的。

值得注意的是，objective函数返回的是accuracy，讲道理，搜索参数的目标是为了最大化该分类任务的accuracy，所以在创建study object的时候指定了direction为"maximize"。如果定义objective函数时返回的是类似error rate的值，则应该将direction指定为"minimize"。

```python
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

这里就是一些optuna的调用代码。

## 自己使用optuna的思路

1）sample个靠谱的子数据集；

2）大概写个objective函数的训练和测试代码，objective函数返回一个需要优化的metric；

3）把要优化的变量定义成optuna的parameter（通过trial.suggest_xxx)；

4）copy个main部分代码，开始搜超参；