from sklearn.svm import SVC
from Code.evaluation.predict import predict_score
from Code.preprocessing.preprocess_fallrisk import preprocess
from Code.plotting.plots import plot_barh
from sklearn.inspection import permutation_importance
from  sklearn.metrics import classification_report
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
    disp = skplt_m.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap = cmap)
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name +"_CONFUSION.png")


if __name__ == '__main__':
    train_acc = []
    test_acc = []

    name = "svm/SVM_fallrisk_TUGNOSIT_FOLDS_"
    data, labels = preprocess(file='/Code/data\Dataset-fallrisk - TUGNOSIT.xlsx')
    #data, labels = preprocess(file='C:\\Users\\annin\PycharmProjects\Master-Degree-Thesis\Code\data\Dataset-fallrisk - TUGFORMULA.xlsx')
    #name = "svm/SVM_fallrisk_TUGFormula_FOLDS_"
    st = StratifiedKFold(n_splits=3, shuffle=False)
    for i, (train_ind, test_ind) in enumerate(st.split(data, labels, groups=None)):
        train, test = data.iloc[train_ind], data.iloc[test_ind]
        labeltrain, labeltest = labels.iloc[train_ind], labels.iloc[test_ind]
        print(train.to_markdown())
        print(test.to_markdown())
        # Scale data
        train, test = scale(train, test)

        # Remove features with variance <0.01
        train, test = variance(train, test, threshold=0.4)


        #FALL RISK TUG FORMULA
        #svm= SVC(class_weight='balanced', kernel='rbf', C=0.8, degree=3, coef0=10, decision_function_shape='ovo')
        #fall risk no sit
        svm = SVC(class_weight='balanced', kernel='linear', C=0.8, degree=1, coef0=10, decision_function_shape='ovo')
        svm.fit(train, labeltrain)

        train_acc.append(predict_score(svm, train, labeltrain))
        test_acc.append(predict_score(svm, test, labeltest))

        # Performance
        print("Train Accuracy: ", predict_score(svm, train, labeltrain))
        print("Test Accuracy: ", predict_score(svm, test, labeltest))

        confusion_matrix(svm, test, labeltest, 'Oranges', name=name + str(i))
        perm_importance = permutation_importance(svm, train, labeltrain)

        # Making the sum of feature importance being equal to 1.0,
        # so feature importance can be understood as percentage
        perm_importance_normalized = perm_importance.importances_mean / perm_importance.importances_mean.sum()

        # Feature's name (considering your X a DataFrame)
        feature_names = train.columns

        sorted_idx = np.argsort(perm_importance_normalized)[::-1][:15]
        features = np.array(feature_names)[sorted_idx]
        plot_barh(features, np.sort(perm_importance_normalized)[::-1], 'chocolate', name + str(i))

        print(classification_report(labeltest, svm.predict(test), digits=2, output_dict=False, zero_division='warn'))

    print("On 4 Folds:\nAverage Train: ", str(np.mean(np.array(train_acc))) + "\tSTD: " + str(np.std(np.array(train_acc))),
          "\nAverage Test: ",str(np.mean(np.array(test_acc))) + "\tSTD: " + str(np.std(np.array(test_acc))) )

