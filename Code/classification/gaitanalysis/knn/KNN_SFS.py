from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from Code.preprocessing.preprocess import preprocess
from Code.plotting.plots import confusion_matrix
if __name__ == '__main__':

        name = "knn/KNN_SFS"
        train, test, labeltrain, labeltest = preprocess()

        for i in range(1,len(train.columns)):
            clf_gridcv = KNeighborsClassifier(algorithm='auto', metric='minkowski', n_neighbors=1, p=2, weights='uniform', n_jobs=-1)
            sfs = SequentialFeatureSelector(clf_gridcv,n_features_to_select=i, tol=None, direction='forward', scoring='balanced_accuracy', cv=10,
                                            n_jobs=-1)
            print(sfs.fit(train, labeltrain))

            clf_gridcv.fit(sfs.transform(train), labeltrain)
            print(sfs.get_feature_names_out(train.columns))
            print(clf_gridcv.score(sfs.transform(test), labeltest))

            # Write to File
            file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/knn/knn_sfs.txt", "a")
            file.write("\nModel " + name + str(i) + '\n')
            file.write("Train Accuracy: " + str(clf_gridcv.score(sfs.transform(train), labeltrain)) + '\n')
            file.write("Test Accuracy: " + str(clf_gridcv.score(sfs.transform(test), labeltest)) + '\n')
            file.write("Features: " + str(sfs.get_feature_names_out(train.columns)) + '\n')
            file.close()
