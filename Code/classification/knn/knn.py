from sklearn.neighbors import KNeighborsClassifier
from Code.preprocessing.preprocess import preprocess
from Code.plotting.plots import confusion_matrix
import numpy as np

if __name__ == '__main__':

        name = "knn/KNN_last"
        train, test, labeltrain, labeltest = preprocess()

        clf = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform', algorithm = 'auto', p = 2, metric = 'minkowski', n_jobs=-1)

        clf.fit(train, labeltrain)

        print(clf.predict(test))
        print(list(labeltest))
        print(clf.score(test, labeltest))
        confusion_matrix(clf, test, labeltest, 'Greens', name=name)

        he = clf.predict(test)
        y_test = np.asarray(labeltest)
        for i in range(test.shape[0]):
            print(y_test[i], he[i])



