from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,balanced_accuracy_score
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance
from Code.preprocessing.preprocess_fallrisk import preprocess_folds
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import sklearn.metrics as skplt_m
import os
from sklearn.model_selection import StratifiedGroupKFold
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
import pandas as pd

if __name__ == '__main__':

    # Number of folds & task
    n_folds = 3

    # Lists to store values for each Fold
    mean_train = []
    mean_test = []
    total_train = []
    total_test = []


    best_test = {}
    best_test['Seed'] = 0
    best_test['Train Accuracy'] = 0.0
    best_test['Test Accuracy'] = 0.0

    data, labels = preprocess_folds()

    seeds = np.random.randint(0, 9999999, size = 100)
    for seed in seeds:
        name = "RandomForest_fall_risk_SEEDS_" + str(seed) + "_" + str(n_folds)+"FOLDS_"

        st = StratifiedGroupKFold(n_splits=n_folds, shuffle=False)

        for i, (train_ind, test_ind) in enumerate(st.split(data, labels, groups=data['Patient'])):
            train, test = data.iloc[train_ind], data.iloc[test_ind]
            labeltrain, labeltest = labels.iloc[train_ind], labels.iloc[test_ind]

            # Scale data
            train, test = scale(train, test)
            # Remove features with variance <0.4
            train, test = variance(train, test, threshold=0.4)

            clf = RandomForestClassifier(criterion='gini', max_depth=3, max_features='sqrt', min_samples_split=2,min_samples_leaf=1, n_estimators=70, random_state=seed,class_weight='balanced', bootstrap = False)
            clf.fit(train, labeltrain)

            train_score = predict_score(clf, train, labeltrain)
            test_score = predict_score(clf, test, labeltest)

            mean_train.append(train_score)
            mean_test.append(test_score)
            total_train.append(train_score)
            total_test.append(test_score)

            # Print Performance
            print("Train Accuracy: ", train_score)
            print("Test Accuracy: ", test_score)
            print(classification_report(labeltest, clf.predict(test), digits=3, output_dict=False, zero_division='warn'))

        print("On 3 Folds: " +
            "\nAverage Train: ", str(np.mean(np.array(mean_train))) +"\tSTD: " + str(np.std(np.array(mean_train))),
            "\nAverage Test: ", str(np.mean(np.array(mean_test))) +"\tSTD: " + str(np.std(np.array(mean_test))))


        if np.mean(np.array(mean_test)) > best_test['Test Accuracy']:
            best_test['Seed'] = seed
            best_test['Train Accuracy'] = np.mean(np.array(mean_train))
            best_test['Test Accuracy'] = np.mean(np.array(mean_test))

        train_acc = []
        test_acc = []

    print("Average on all folds for all seeds:")
    print("Train Accuracy: ", np.mean(np.array(total_train)), "std: ", np.std(np.array(total_train)))
    print("Test Accuracy: ", np.mean(np.array(total_test)),"std: ",np.std(np.array(total_test)))
    print("best", str(best_test))