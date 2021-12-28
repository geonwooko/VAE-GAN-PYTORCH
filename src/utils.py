from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def log_param(param):
    for key, value in param.items():
        logger.info(f"{key} : {value}")









