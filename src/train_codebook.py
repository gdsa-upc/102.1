# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
from scipy.cluster.vq import kmeans,vq
import numpy as np

def train_codebook(nclusters,normalized_descriptors):
    return kmeans(normalized_descriptors,nclusters) #obtenim els centroides de les imatges