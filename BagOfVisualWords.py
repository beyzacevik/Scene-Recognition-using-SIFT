import numpy as np
import pandas as pd

import cv2
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler


class BagOfVisualWords(object):

    def __init__(self, image_list):
        self.image_list = image_list

    def extract_sift_features(self):

        sift = cv2.SIFT_create()
        images_descriptors = dict()
        for image_path in self.image_list:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptor = sift.detectAndCompute(gray, None)
            images_descriptors[image_path] = descriptor

        return images_descriptors

    def build_vocabulary(self, descriptors, vocab_size):

        k_means = KMeans(n_clusters=vocab_size, random_state=0).fit(descriptors)
        centers = k_means.cluster_centers_.tolist()
        closest, _ = pairwise_distances_argmin_min(centers, descriptors)
        vocabulary = descriptors[closest]

        return vocabulary

    def create_histograms(self, images_descriptors, vocabulary, vocab_size):

        histograms = np.zeros((len(self.image_list), vocab_size), dtype="float64")
        visuals_words_and_distances = [vq(images_descriptors[img], vocabulary) for img in self.image_list]
        for i, word_distance in enumerate(visuals_words_and_distances):
          for j in word_distance[0]:
              histograms[i][j] += 1

        return histograms

    def convert_to_ndarray(self, descriptors):

        descriptors_array = descriptors[0]
        for descriptor in descriptors[1:]:
          descriptors_array = np.concatenate((descriptors_array, descriptor), axis=0)

        return descriptors_array

    def convert_features_to_df(self, features):

        df = pd.DataFrame(features)
        img_idx = pd.Series([i for i in range(len(self.image_list))])
        df.set_index(img_idx)

        return df

    def normalize(self, features):

        scaler = MinMaxScaler().fit(features)
        normalized_features = scaler.transform(features)

        return normalized_features