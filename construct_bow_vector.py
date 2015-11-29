#-*- coding: utf-8 -*-    import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from train_kmeans import train_codebook
from get_local_features import get_local_features
from compute_assignments import get_assignments

def build_bow(train_codebook,id_image):
    