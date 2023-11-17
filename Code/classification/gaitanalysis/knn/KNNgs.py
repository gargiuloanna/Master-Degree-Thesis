from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from Code.preprocessing.preprocess import preprocess
from Code.plotting.plots import confusion_matrix

if __name__ == '__main__':

        name = "knn/KNN_GS_GAITPATIENT"
        train, test, labeltrain, labeltest = preprocess()

        param_grid = {
            'n_neighbors': [1, 2, 3, 4, 5, 6,7 ,8 ,9 , 10, 11,12],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 4, 6, 9],
            'metric': ['minkowski']
        }
        clf = KNeighborsClassifier(n_jobs=-1)

        clf_gridcv = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring='balanced_accuracy', error_score=0, n_jobs=-1, verbose=3, refit=True)
        clf_gridcv.fit(train, labeltrain)

        print(f"Best KNN with params: {clf_gridcv.best_params_} and score {clf_gridcv.best_score_}")

        clf = clf_gridcv.best_estimator_

        # Write to File
        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/knn/knn_gs.txt", "a")
        file.write("\nModel " + name + '\n')
        file.write(f"Best KNN with params: {clf_gridcv.best_params_} and score {clf_gridcv.best_score_}")
        file.write("Train Accuracy: " + str(clf.score(train, labeltrain)) + '\n')
        file.write("Test Accuracy: " + str(clf.score(test, labeltest)) + '\n')
        file.close()

        print(clf.predict(test))
        print(list(labeltest))
        print(clf.score(test, labeltest))
        confusion_matrix(clf, test, labeltest, 'Greens', name=name)



