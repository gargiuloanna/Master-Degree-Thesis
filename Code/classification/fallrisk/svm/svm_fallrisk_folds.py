import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sklearn.metrics as skplt_m
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from Code.preprocessing.biological_data import get_age
from Code.preprocessing.scale import scale
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.preprocess_fallrisk import preprocess_folds
from Code.evaluation.predict import predict_score
from Code.plotting.plots import plot_barh

if __name__ == '__main__':

    #Number of folds & task
    n_folds = 5
    task = 'TUG'

    #Lists to store values for each Fold
    train_acc = []
    test_acc = []
    f1_score = []
    precision = []
    recall = []

    #Name of the classifier
    name = "svm/SVM_fallrisk_"+str(n_folds)+"FOLDS_" + task

    data, labels = preprocess_folds()

    feature_import = {}
    for colum in data.columns:
        feature_import[colum] = 0.0

    st = StratifiedGroupKFold(n_splits=n_folds)

    for i, (train_ind, test_ind) in enumerate(st.split(data, labels, groups=data['Patient'])):
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

        svm = SVC(class_weight='balanced', kernel='poly', C=0.1, degree=3, coef0=10, decision_function_shape='ovo')
        svm.fit(train, labeltrain)

        #get stats
        accuracy_train =predict_score(svm, train, labeltrain)
        accuracy_test =predict_score(svm, test, labeltest)
        prec, rec, f1, _ = skplt_m.precision_recall_fscore_support(labeltest, svm.predict(test), beta=1.0, pos_label=1, average='weighted',warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division='warn')
        to_print['Predicted'] = svm.predict(test)

        #Append to lists
        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)
        f1_score.append(f1)
        precision.append(prec)
        recall.append(rec)

        # Print Performance
        print("Train Accuracy: ", accuracy_train)
        print("Test Accuracy: ", accuracy_test)
        print(skplt_m.classification_report(labeltest, svm.predict(test), digits=2, output_dict=False, zero_division='warn'))
        print(to_print.to_markdown())

        #get feature importance
        perm_importance = permutation_importance(svm, train, labeltrain)
        perm_importance_normalized = perm_importance.importances_mean / perm_importance.importances_mean.sum()
        feature_names = train.columns
        sorted_idx = np.argsort(perm_importance_normalized)[::-1]
        features = np.array(feature_names)[sorted_idx]
        for i in range(len(feature_names)):
            feature_import[features[i]] += np.sort(perm_importance_normalized)[::-1][i]

    print("On " + str(n_folds) + " Folds:" +
        "\nAverage Train: ", str(np.mean(np.array(train_acc))) + "\tSTD: " + str(np.std(np.array(train_acc))),
        "\nAverage Test: ", str(np.mean(np.array(test_acc))) + "\tSTD: " + str(np.std(np.array(test_acc))),
        "F1 Score: " + str(np.mean(np.array(f1_score))) + "\tSTD: " + str(np.std(np.array(f1_score))),
        "Precision: " + str(np.mean(np.array(precision))) + "\tSTD: " + str(np.std(np.array(precision))),
        "Recall: " + str(np.mean(np.array(recall))) + "\tSTD: "+ str(np.std(np.array(recall)))
          )

    #plot most important on all folds
    fd = pd.DataFrame(data=feature_import.items(), columns=['Feature', 'Importance'])
    fd.sort_values(by="Importance", axis=0, ascending=False, inplace=True)
    plot_barh(fd['Feature'][:15], fd['Importance'][:15], 'chocolate', name)
