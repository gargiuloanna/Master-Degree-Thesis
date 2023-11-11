from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,balanced_accuracy_score
from Code.evaluation.predict import predict_score
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance
from Code.preprocessing.preprocess_fallrisk import preprocess
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import sklearn.metrics as skplt_m
import os
from sklearn.model_selection import StratifiedKFold
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale

def confusion_matrix(classifier, data, labels, cmap, name = "Model_Confusion"):

    lab_pred = classifier.predict(data)

    plt.figure(figsize = (34,34))
    cm = skplt_m.confusion_matrix(labels, lab_pred)
    disp = skplt_m.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['At Risk', 'Not At Risk'])
    disp.plot(cmap = cmap)
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name +"_CONFUSION.png")

if __name__ == '__main__':

    mean_train = []

    mean_test = []
    best_test = {}
    best_test['Seed'] = 0
    best_test['Train Accuracy'] = 0.0
    best_test['Test Accuracy'] = 0.0
    total_train = []
    total_test = []
    data, labels = preprocess(file='/Code/data\Dataset-negativeselection - TUGNOSIT.xlsx')

    seeds = np.random.randint(0, 9999999, size = 100)
    for seed in seeds:
        name = "RandomForest_fall_risk_tugNOSIT" + str(seed)

        st = StratifiedKFold(n_splits=3, shuffle=False)
        for i, (train_ind, test_ind) in enumerate(st.split(data, labels, groups=None)):
            train, test = data.iloc[train_ind], data.iloc[test_ind]
            labeltrain, labeltest = labels.iloc[train_ind], labels.iloc[test_ind]

            # Scale data
            train, test = scale(train, test)

            # Remove features with variance <0.4
            train, test = variance(train, test, threshold=0.4)

            #TUG FORMULA
            '''
            clf = RandomForestClassifier(criterion='entropy', max_depth=3, max_features='sqrt', min_samples_split=2,
                                     min_samples_leaf=0.1, n_estimators=140, random_state=seed,
                                     class_weight='balanced', bootstrap = False)
                                     
            #Average performance on set is 80 on test and 80.7 on oob%
            '''
            #TUG NO SIT
            clf = RandomForestClassifier(criterion='gini', max_depth=3, max_features='sqrt', min_samples_split=2,
                                         min_samples_leaf=1, n_estimators=70, random_state=seed,
                                         class_weight='balanced', bootstrap = False)

            clf.fit(train, labeltrain)

            train_score = predict_score(clf, train, labeltrain)
            test_score = predict_score(clf, test, labeltest)

            mean_train.append(train_score)
            mean_test.append(test_score)

            total_train.append(train_score)
            total_test.append(test_score)

        print("On 4 Folds:\nAverage Train: ", str(np.mean(np.array(mean_train))) +
              "\tSTD: " + str(np.std(np.array(mean_train))),
              "\nAverage Test: ", str(np.mean(np.array(mean_test))) +
              "\tSTD: " + str(np.std(np.array(mean_test))))


        if np.mean(np.array(mean_test)) > best_test['Test Accuracy']:
            best_test['Seed'] = seed
            best_test['Train Accuracy'] = np.mean(np.array(mean_train))
            best_test['Test Accuracy'] = np.mean(np.array(mean_test))

            # Performance
            print("Train Accuracy: ", predict_score(clf, train, labeltrain))
            print("Test Accuracy: ", predict_score(clf, test, labeltest))


            print(classification_report(labeltest, clf.predict(test),  digits=3, output_dict=False, zero_division='warn'))

        train_acc = []
        test_acc = []

    print("Average on all folds for all seeds:")
    print("Train Accuracy: ", np.mean(np.array(total_train)), "std: ", np.std(np.array(total_train)))
    print("Test Accuracy: ", np.mean(np.array(total_test)),"std: ",np.std(np.array(total_test)))
    print("best", str(best_test))