from get_local_features import get_local_features
import os
import cv2
import numpy as np
import sklearn.metrics
from sklearn.metrics import kmeans

def train_codebook(normalize_descriptors):
    