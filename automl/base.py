# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import random
import inspect
import warnings
import sys
import imp
from functools import partial
from datetime import datetime
from multiprocessing import cpu_count
import errno

from tempfile import mkdtemp
from shutil import rmtree

import numpy as np
from pandas import DataFrame
from scipy import sparse
from pyspark.ml.base import




class AutoBase(BaseEstimator):
    """Automatically creates and optimizes machine learning pipelines using GP."""

    classification = None  # set by child classes
