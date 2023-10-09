import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
import pandas as pd
from Code.preprocessing.scale import scale
from Code.plotting.plots import confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
if __name__ == '__main__':

        df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-only normalized lengths.xlsx')
        df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

        data = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
        train, test = scale(train, test)

        param_grid = {
            'n_neighbors': [1, 2, 3, 4, 5, 6,7 ,8 ,9 , 10, 11,12],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 4, 6, 9],
            'metric': ['minkowski']
        }

        for i in range(1,len(data.columns)):
            clf_gridcv = KNeighborsClassifier(algorithm='auto', metric='minkowski', n_neighbors=1, p=2, weights='uniform', n_jobs=-1)
            sfs = SequentialFeatureSelector(clf_gridcv,n_features_to_select=i, tol=None, direction='forward', scoring='balanced_accuracy', cv=10,
                                            n_jobs=-1)
            print(sfs.fit(train, labeltrain))
            clf_gridcv.fit(sfs.transform(train), labeltrain)
            print(sfs.get_feature_names_out(data.columns))
            print(clf_gridcv.score(sfs.transform(test), labeltest))
        '''
        #clf_gridcv = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring='balanced_accuracy', error_score=0, n_jobs=-1, verbose=3, refit=True)
        #clf_gridcv.fit(train, labeltrain)
        #print(f"Best KNN with params: {clf_gridcv.best_params_} and score {clf_gridcv.best_score_}")
        #clf = clf_gridcv.
        clf = clf_gridcv

        print(clf.predict(test))
        print(list(labeltest))
        print(clf.score(test, labeltest))
        if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + "KNN"):
            os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + "KNN")
        #confusion_matrix(clf, test, labeltest, name="KNN")

        for i in range(0, len(data.columns) -1):
            plt.scatter(data.iloc[:,i], data.iloc[:,i+1], c=labels, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"]), edgecolor="k", s=20)
            plt.xlabel(data.columns[i])
            plt.ylabel(data.columns[i+1])
            plt.title("{} (k = {})".format("KNN", 1))
            plt.text(
                0.9,
                0.1,
                "{:.2f}".format(clf.score(test, labeltest)),
                size=15,
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

            plt.show()
        '''

