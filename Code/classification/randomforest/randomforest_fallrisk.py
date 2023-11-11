from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Code.evaluation.predict import predict_score
from Code.plotting.plots import  permutation_imp, feature_importance
from Code.preprocessing.preprocess_fallrisk import preprocess
import numpy as np
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


    #TUG FORMULA
    seed_formula = 9161817

    #TUG NO SIT
    seed_nosit = 8358684

    '''
    seed = seed_formula
    name = "RandomForest_fall_risk_tugFORMULA" + str(seed)
    data, labels = preprocess(file='C:\\Users\\annin\PycharmProjects\Master-Degree-Thesis\Code\Data\Dataset-fallrisk - TUGFORMULA.xlsx')
    '''
    seed = seed_nosit
    name = "RandomForest_fall_risk_tugNOSIT" + str(seed)
    data, labels = preprocess(file='\Code\data\Dataset-fallrisk - TUGNOSIT.xlsx')


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
                               min_samples_leaf=0.1, n_estimators=140, random_state=seed_formula,
                               class_weight='balanced', bootstrap = False)
        '''
        #NO SIT
        clf = RandomForestClassifier(criterion='gini', max_depth=3, max_features='sqrt', min_samples_split=2,
                                           min_samples_leaf=1, n_estimators=70, random_state=seed_nosit,
                                           class_weight='balanced', bootstrap = False)

        clf.fit(train, labeltrain)

        train_score = predict_score(clf, train, labeltrain)
        test_score = predict_score(clf, test, labeltest)

        mean_train.append(train_score)
        mean_test.append(test_score)
        # Performance
        print("Train Accuracy: ", predict_score(clf, train, labeltrain))
        print("Test Accuracy: ", predict_score(clf, test, labeltest))

        print(classification_report(labeltest, clf.predict(test), digits=3, output_dict=False, zero_division='warn'))

        # Plots
        confusion_matrix(clf, test, labeltest, 'Blues', name=name + str(i))
        feature_importance(clf, train.columns, 'skyblue', name=name+ str(i))
        permutation_imp(clf, test, labeltest, 'skyblue', name=name+ str(i))

    print("On 4 Folds:\nAverage Train: ", str(np.mean(np.array(mean_train))) +
              "\tSTD: " + str(np.std(np.array(mean_train))),
              "\nAverage Test: ", str(np.mean(np.array(mean_test))) +
              "\tSTD: " + str(np.std(np.array(mean_test))))