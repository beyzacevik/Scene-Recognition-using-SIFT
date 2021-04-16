import os


class Dataset(object):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def extract_dataset(self):

        images = []
        labels = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".jpg"):
                    label = subdir.split('/')[-1]
                    images.append(filepath)
                    labels.append(label)

        return images, labels




