from SceneRecognition.TinyImage import *
from SceneRecognition.BagOfVisualWords import *
from SceneRecognition.LinearClassifier import *
from SceneRecognition.Dataset import *
from SceneRecognition.VisualizationUtils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    obj = Dataset()
    images, labels = obj.extract_dataset()

    bovw = BagOfVisualWords(images)
    images_descriptors = bovw.extract_sift_features()
    descriptors = bovw.fetch_only_features(images_descriptors)
    vocabulary = bovw.build_vocabulary(descriptors=descriptors, vocab_size=200)
    histograms = bovw.create_histograms(images_descriptors, vocabulary, 200)

    scaler = MinMaxScaler().fit(histograms)
    histograms = scaler.transform(histograms)

    histogram_df = bovw.convert_features_to_df(histograms)

    x_train_df, x_test_df, y_train, y_test = train_test_split(histogram_df, labels, random_state=42, shuffle=True, test_size=0.20)
    x_train = np.asarray(x_train_df)
    x_test = np.asarray(x_test_df)

    clf = LinearClassifier(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    knn_predicted_sift = clf.knn_classifier()
    svm_predicted_sift = clf.svm_classifier()


    tinyImage = TinyImage(images)
    tinyImage_features = tinyImage.extract_tiny_image_features()
    tinyImage_df = bovw.convert_features_to_df(tinyImage_features)

    x_train_df, x_test_df, y_train, y_test = train_test_split(tinyImage_df, labels, random_state=42, shuffle=True, test_size=0.20)
    x_train = np.asarray(x_train_df)
    x_test = np.asarray(x_test_df)

    clf = LinearClassifier(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    knn_predicted_tinyImage = clf.knn_classifier()
    svm_predicted_tinyImage = clf.svm_classifier()

'''an example of displaying image it can be reproduced for each experiment not all shown for the sake of simplicity'''
    # f = plt.figure(figsize=(12, 12))
    # axs = f.subplots(5, 6)
    #
    # pos_bedrooms, pos_office, pos_highway, pos_kitchen, pos_livingroom, pos_mountains = get_successful_images(y_test, knn_predicted_sift, x_test_df, images)
    # show_images(pos_mountains, images, 'Mountains', f, axs, 0)
    # show_images(pos_highway, images, 'Highway', f, axs, 1)
    # show_images(pos_office, images, 'Office', f, axs, 2)
    # show_images(pos_livingroom, images, 'LivingRoom', f, axs, 3)
    # show_images(pos_kitchen, images, 'Kitchen', f, axs, 4)
    # show_images(pos_bedrooms, images, 'Bedroom', f, axs, 5)


