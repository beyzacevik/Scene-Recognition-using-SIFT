import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler


class TinyImage(object):

    def __init__(self, image_list, size=16):
        self.image_list = image_list
        self.dim = (size, size)

    def extract_tiny_image_features(self):

        tiny_images = []
        for image_path in self.image_list:
            image = cv2.imread(image_path, 0)
            image.resize(self.dim)
            scaler = MinMaxScaler()
            scaler.fit_transform(image)
            image = image.flatten()
            tiny_images.append(image)

        return np.asarray(tiny_images)


