import copy
from itertools import product
import numpy as np

from .base import AutoBase
from .config.classifier import classifier_config_dict
from .config.regressor import regressor_config_dict



class AutoClassifier(AutoBase):
    """TPOT estimator for classification problems."""

    scoring_function = 'accuracy'  # Classification scoring
    default_config_dict = classifier_config_dict  # Classification dictionary
    classification = True
    regression = False

    def _init_pretest(self, dataset):
        """Set the sample of data used to verify pipelines work
        with the passed data set.

        This is not intend for anything other than perfunctory dataset
        pipeline compatibility testing
        """
        NotImplemented


class TPOTRegressor(AutoBase):
    """TPOT estimator for regression problems."""

    scoring_function = 'mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True

    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work with the passed data set.

        """
        NotImplemented