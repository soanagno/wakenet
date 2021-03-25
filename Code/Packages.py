# Packages

import os
import time
import json
import random
import warnings
import numpy as np
import scipy.stats as stats

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import dask
from dask.distributed import Client, progress

from scipy import ndimage
from scipy import interpolate
from scipy.optimize import minimize

import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.yaw import YawOptimization
from floris.tools.optimization.scipy.layout import LayoutOptimization
import logging
logging.getLogger("floris").setLevel(logging.WARNING)