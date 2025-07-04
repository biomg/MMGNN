import argparse
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from utils_fp_downstream import *
from torch_geometric.loader import DataLoader
from encoder_gnn import GINNet, GATNet
from model_gnn_fp_downstream import Model_gat_fp, Model_gin_fp, Model_gnn_fp
import torch.nn as nn
import math
import os
import numpy as np
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)
import random
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
lambda_contrast = 0.05  # 初始对比损失权重
decay_rate = 0.97      # 衰减率


task = 'bbbp'
path = 'down_task'



if __name__ == '__main__':


    train_data = TestbedDataset(root=path, dataset='train', task=task)
    valid_data = TestbedDataset(root=path, dataset='valid', task=task)
    test_data = TestbedDataset(root=path, dataset='test', task=task)