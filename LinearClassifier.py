from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns


class LinearClassifier(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classes = ['Bedroom','Highway', 'Kitchen','LivingRoom','Mountain', 'Office']

    def svm_classifier(self):
        svc_model = SVC()
        svc = svc_model.fit(self.x_train, self.y_train)
        svc_predicted = svc.predict(self.x_test)
        print('SVM Accuracy: ', str(self.calculate_accuracy(svc_predicted)))
        print('***SVM Classification Results***\n', classification_report(self.y_test, svc_predicted))
        #   self.create_confusion_matrix(self, svc_predicted, SVM')
        return svc_predicted

    def knn_classifier(self):
        knn_model = KNeighborsClassifier().fit(self.x_train, self.y_train)
        knn_predicted = knn_model.predict(self.x_test)
        print('KNN Accuracy: ', str(self.calculate_accuracy(knn_predicted)))
        print('***KNN Classification Results***\n', classification_report(self.y_test, knn_predicted))
        #  self.create_confusion_matrix(self,knn_predicted, 'KNN')
        return knn_predicted

    def calculate_accuracy(self, y_pred):
        # accuracy TP+TN / TP+TN+FP+FN == correct/all
        correct = 0
        for pred, real in zip(y_pred,self.y_test):
            if pred == real:
                correct += 1
        acc = correct / float(len(y_pred)) * 100.0

        return acc

    def create_confusion_matrix(self, y_pred, exp):

        fig, ax = plt.subplot(figsize=(16, 16))

        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes)
        sns.heatmap(cm, annot=True, ax=ax, cmap='coolwarm', fmt="d")

        ax.set_title('Confusion Matrix of {}'.format(exp))
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        ax.xaxis.set_ticklabels(self.classes)
        ax.yaxis.set_ticklabels(self.classes)
        plt.show()
        return

