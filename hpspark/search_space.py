import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope, as_apply


def _trees_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))


def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])

def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])


def _grad_boosting_reg_loss_alpha(name):
    return hp.choice(name, [
        ('ls', 0.9),
        ('lad', 0.9),
        ('huber', hp.uniform(name + '.alpha', 0.85, 0.95)),
        ('quantile', 0.5)
    ])


def _grad_boosting_learning_rate(name):
    return hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))


def _grad_boosting_subsample(name):
    return hp.pchoice(name, [
        (0.2, 1.0),  # default choice.
        (0.8, hp.uniform(name + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
    ])



cls_space = hp.choice('classifier_type', [
    {
        'type': 'decision_tree',
        'maxDepth' : _trees_max_depth('decision_tree.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('decision_tree.minInstancesPerNode')
    },
    {
        'type': 'randomforest',
        'subsamplingRate': _grad_boosting_subsample('randomforest.subsamplingRate'),
        'numTrees' :  _trees_n_estimators('randomforest.numTrees'),
        'maxDepth': _trees_max_depth('randomforest.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('randomforest.minInstancesPerNode')
    },
    {
        'type': 'grad_boosting',
        'stepSize': _grad_boosting_learning_rate('grad_boosting.stepSize'),
        'subsamplingRate': _grad_boosting_subsample('grad_boosting.subsamplingRate'),
        'maxDepth': _trees_max_depth('grad_boosting.maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('grad_boosting.minInstancesPerNode')
    }
])


reg_space = hp.choice('regresssior_type', [
    {
        'type': 'decision_tree',
        'maxDepth' : _trees_max_depth('maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('minInstancesPerNode')
    },
    {
        'type': 'randomforest',
        'subsamplingRate': _grad_boosting_subsample('subsamplingRate'),
        'numTrees' :  _trees_n_estimators('numTrees'),
        'maxDepth': _trees_max_depth('maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('minInstancesPerNode')
    },
    {
        'type': 'grad_boosting',
        'stepSize': _grad_boosting_learning_rate('stepSize'),
        'subsamplingRate': _grad_boosting_subsample('subsamplingRate'),
        'maxDepth': _trees_max_depth('maxDepth'),
        'minInstancesPerNode': _trees_min_samples_leaf('minInstancesPerNode')
    }
])