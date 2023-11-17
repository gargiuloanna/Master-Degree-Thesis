import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skplt_m
from sklearn.model_selection import StratifiedGroupKFold
from Code.preprocessing.biological_data import get_age
from Code.preprocessing.scale import scale
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.preprocess_fallrisk import preprocess_folds
from Code.evaluation.predict import predict_score
from Code.plotting.plots import plot_barh

if __name__ == '__main__':

    # Number of folds
    n_folds = 3


    # Lists to store values for each Fold
    mean_train = []
    mean_test = []

    # Name of the classifier
    seed = 2388936
    name = "RandomForest_GAIT_TASK_" + str(seed) + "_" + str(n_folds)+"FOLDS"


    st = StratifiedGroupKFold(n_splits=n_folds)

    data, labels = preprocess_folds()
    feature_import = {}
    for colum in data.columns:
        feature_import[colum] = 0.0

    for i, (train_ind, test_ind) in enumerate(st.split(data, labels, groups=data['Exercise'])):
        train, test = data.iloc[train_ind], data.iloc[test_ind]
        labeltrain, labeltest = labels.iloc[train_ind], labels.iloc[test_ind]

        # get patients
        sav = []
        for patient in test['Patient']:
            sav.append(get_age(patient))

        to_print = pd.DataFrame()
        to_print['Patient'] = test['Patient']
        to_print['Task'] = test['Exercise']
        to_print['Age'] = sav
        to_print['Label'] = labeltest

        # Scale data
        train, test = scale(train, test)
        # Remove features with variance <0.4
        train, test = variance(train, test, threshold=0.4)

        clf = RandomForestClassifier(criterion='entropy', max_depth=3, max_features='sqrt', min_samples_split=10, min_samples_leaf=1, n_estimators=140, random_state=seed, oob_score=balanced_accuracy_score, class_weight='balanced')
        clf.fit(train, labeltrain)

        # get stats
        accuracy_train = predict_score(clf, train, labeltrain)
        accuracy_test = predict_score(clf, test, labeltest)
        to_print['Predicted'] = clf.predict(test)

        # Append to lists
        mean_train.append(accuracy_train)
        mean_test.append(accuracy_test)

        # Print Performance
        print("Train Accuracy: ", accuracy_train)
        print("Test Accuracy: ", accuracy_test)
        print(skplt_m.classification_report(labeltest, clf.predict(test), digits=3, output_dict=False, zero_division='warn'))
        print(to_print.to_markdown())

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        feature_names = np.array(train.columns)[indices]
        for i in range(len(feature_names)):
            feature_import[feature_names[i]] += np.sort(importances)[::-1][i]

    print("On " + str(n_folds) + " Folds:" +
          "\nAverage Train: ", str(np.mean(np.array(mean_train))) + "\tSTD: " + str(np.std(np.array(mean_train))),
          "\nAverage Test: ", str(np.mean(np.array(mean_test))) + "\tSTD: " + str(np.std(np.array(mean_test))),
          )

    # plot most important on all folds
    fd = pd.DataFrame(data=feature_import.items(), columns=['Feature', 'Importance'])
    fd.sort_values(by="Importance", axis=0, ascending=False, inplace=True)
    plot_barh(fd['Feature'][:15], fd['Importance'][:15], 'skyblue', name)

