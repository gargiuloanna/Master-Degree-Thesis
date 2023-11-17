import numpy as np
import pandas as pd
import sklearn.metrics as skplt_m
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SequentialFeatureSelector
from Code.preprocessing.preprocess import preprocess_folds
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
from Code.preprocessing.biological_data import get_age
from Code.evaluation.predict import predict_score


if __name__ == '__main__':

    # Number of folds
    n_folds = 3

    # Lists to store values for each Fold
    mean_train = []
    mean_test = []

    name = "knn/KNN_gait_" + str(n_folds) + "folds"

    data, labels = preprocess_folds()

    st = StratifiedGroupKFold(n_splits=n_folds)


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

            clf = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)
            sfs = SequentialFeatureSelector(clf, n_features_to_select=15, tol=None, direction='forward', scoring='balanced_accuracy', cv=10,n_jobs=-1)

            sfs.fit(train, labeltrain)
            clf.fit(sfs.transform(train), labeltrain)

            #get stats
            train_score = predict_score(clf, sfs.transform(train), labeltrain)
            test_score = predict_score(clf, sfs.transform(test), labeltest)
            to_print['Predicted'] = clf.predict(sfs.transform(test))

            #append to lists
            mean_train.append(train_score)
            mean_test.append(test_score)

            # print Performance
            print("Train Accuracy: ", predict_score(clf, sfs.transform(train), labeltrain))
            print("Test Accuracy: ", predict_score(clf, sfs.transform(test), labeltest))
            print(skplt_m.lassification_report(labeltest, clf.predict(sfs.transform(test)), digits=3, output_dict=False, zero_division='warn'))
            print(to_print.to_markdown())


            file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/knn/knn_sfs_fold.txt", "a")
            file.write("\n\nModel " + name + '\n')
            file.write("Average Train Accuracy: " + str(np.mean(np.array(mean_train))) + '\n')
            file.write("Average Test Accuracy: " + str(np.mean(np.array(mean_test))) + '\n')
            file.write("Features: " + str(np.sort((sfs.get_feature_names_out(train.columns)))) + '\n')
            file.close()






