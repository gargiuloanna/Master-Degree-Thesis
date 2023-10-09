from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from Code.evaluation.predict import predict_score
from joblib import dump
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance
from Code.preprocessing.scale import scale
from  sklearn.metrics import balanced_accuracy_score
import os

if __name__ == '__main__':
    #['Dataset -features_changed.xlsx','Dataset -features_changed - lr.xlsx','Dataset -features_changed - mean.xlsx' 'Dataset -features_changed.xlsx','Dataset -features_changed - lr.xlsx','Dataset -features_changed - mean.xlsx' ,
    #'Dataset-normalized lengths.xlsx', 'Dataset-normalized lengths - lr.xlsx','Dataset-normalized lengths - mean.xlsx',,'Dataset-normalized lengths.xlsx', 'Dataset-normalized lengths - lr.xlsx','Dataset-normalized lengths - mean.xlsx','Dataset-only normalized lengths.xlsx','Dataset-only normalized lengths -lr.xlsx','Dataset-only normalized lengths - mean.xlsx']
    for i in ['Dataset-only normalized lengths.xlsx']:
        name = "RandomForestDifferentDatasets30%10fold - " + i
        df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/' + i)
        df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

        data = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        train, test, labeltrain, labeltest =train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
        train, test = scale(train, test)

        print("[RANDOM FOREST] Searching best params with GridSearchCV")
        rdf_model = RandomForestClassifier(random_state=0, oob_score=balanced_accuracy_score, class_weight='balanced')
        param_grid = {
            'n_estimators': [30, 40, 50, 60, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [0.1, 0.23, 2, 10, 20],
            'min_samples_leaf': [0.1, 0.23, 1, 10, 20],
            'max_features': ['sqrt', 'log2']
        }
        rdf_gridcv = GridSearchCV(rdf_model, param_grid=param_grid, cv=10, scoring='balanced_accuracy',  error_score=0, n_jobs=-1, verbose=3, refit=True)
        rdf_gridcv.fit(train, labeltrain)

        print(f"[RANDOM FOREST] Best random forest with params: {rdf_gridcv.best_params_}")
        # save model
        clf = RandomForestClassifier(n_estimators= rdf_gridcv.best_params_['n_estimators'], criterion = rdf_gridcv.best_params_['criterion'] ,max_depth= rdf_gridcv.best_params_['max_depth'],
                                     min_samples_split= rdf_gridcv.best_params_['min_samples_split'], min_samples_leaf= rdf_gridcv.best_params_['min_samples_leaf'], max_features=rdf_gridcv.best_params_['max_features'],
                                     random_state = 0, oob_score = balanced_accuracy_score, class_weight='balanced')

        clf.fit(train, labeltrain)
        print("Training OOB Accuracy ", clf.oob_score_)
        rdf = dump(rdf_gridcv.best_estimator_, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/models/" + name + ".joblib")
        print("[RANDOM FOREST] Model Saved")

        print("Test Accuracy: ", predict_score(rdf_gridcv.best_estimator_, test, labeltest))
        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/classification/performances.txt", "a")
        file.write("Model " + name + '\n')
        file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
        file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')
        file.close()

        # plots
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + name)
        confusion_matrix(rdf_gridcv.best_estimator_, test, labeltest, name = name)
        feature_importance(rdf_gridcv.best_estimator_, data.columns,name = name)
        random_forest_fimp(rdf_gridcv.best_estimator_, data.columns, name = name)
        permutation_imp(rdf_gridcv.best_estimator_, data, labels, name = name)


