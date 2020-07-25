"""
This file is the overall trainer

Benjamin Keltjens
July 2020
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import os
import pickle

from drone import Drone, SimpleLander
from environment import Environment, Obstacle
from render import Renderer, DataStream
from presets import Presets

import neat
import visualize
