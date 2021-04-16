import numpy as np
import cv2
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd


class BagOfVisualWords(object):

    def __init__(self, image_list):
        self.image_list = image_list

    def extract_sift_features(self):

        sift = cv2.SIFT_create()
        images_descriptors = []
        for image_path in self.image_list:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptor = sift.detectAndCompute(gray, None)
            images_descriptors.append((image_path, descriptor))

        return images_descriptors

    def fetch_only_features(self, images_descriptors):

        features = images_descriptors[0][1]  # extract only the descriptors, eliminate words
        for image_path, feature in images_descriptors[1:]:
            features = np.vstack((features, feature))

        return features

    def build_vocabulary(self, descriptors, vocab_size):

        k_means = KMeans(n_clusters=vocab_size, random_state=0).fit(descriptors)
        centers = k_means.cluster_centers_.tolist()
        closest, _ = pairwise_distances_argmin_min(centers, descriptors)
        vocabulary = descriptors[closest]

        return vocabulary

    def create_histograms(self, images_descriptors, vocabulary, vocab_size):

        num_of_images = len(self.image_list)
        histograms = np.zeros((num_of_images, vocab_size), dtype="float64")

        for img_idx in range(num_of_images):
            visual_words, distance = vq(images_descriptors[img_idx][1], vocabulary)
            for vocab_idx in visual_words:
                histograms[img_idx][vocab_idx] += 1

        return histograms

    def convert_features_to_df(self, features):

        df = pd.DataFrame(features)
        img_idx = pd.Series([i for i in range(len(self.image_list))])
        df.set_index(img_idx)

        return df
