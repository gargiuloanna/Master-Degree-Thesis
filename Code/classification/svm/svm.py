from sklearn.svm import SVC
from Code.evaluation.predict import predict_score
from Code.plotting.plots import plot_barh
from Code.preprocessing.preprocess import preprocess
from  sklearn.metrics import classification_report
import numpy as np
from joblib import dump
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import sklearn.metrics as skplt_m
import os


def confusion_matrix(classifier, data, labels, cmap, name = "Model_Confusion"):

    lab_pred = classifier.predict(data)

    plt.figure(figsize = (34,34))
    cm = skplt_m.confusion_matrix(labels, lab_pred)
    disp = skplt_m.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['Elderly', 'Parkinson', 'Adults'])
    disp.plot(cmap = cmap)
    if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name):
        os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name)

    plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/plots/" + name +"_CONFUSION.png")


if __name__ == '__main__':

    name = "svm/SVM_last"

    train, test, labeltrain, labeltest = preprocess()

    svm = SVC(class_weight='balanced', kernel='rbf',C=2.0, degree=3, coef0=10, decision_function_shape='ovo')


    svm.fit(train, labeltrain)

    # Performance
    print("Train Accuracy: ", predict_score(svm, train, labeltrain))
    print("Test Accuracy: ", predict_score(svm, test, labeltest))


    print(classification_report(labeltest, svm.predict(test), digits=2, output_dict=False, zero_division='warn'))

    # Write to File
    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/svm/svm.txt", "a")
    file.write("\nModel " + name + '\n')
    file.write("Train Accuracy: " + str(predict_score(svm, train, labeltrain)) + '\n')
    file.write("Test Accuracy: " + str(predict_score(svm, test, labeltest)) + '\n')
    file.close()

    # Plots

    rdf = dump(svm, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/" + name + ".joblib")
    print("[SVM] Model Saved")

    # Plots
    print("Dual Coefficients: ", svm.dual_coef_.shape)
    confusion_matrix(svm, test, labeltest, 'Oranges', name=name)
    perm_importance = permutation_importance(svm, train, labeltrain)

    # Making the sum of feature importance being equal to 1.0,
    # so feature importance can be understood as percentage
    perm_importance_normalized = perm_importance.importances_mean / perm_importance.importances_mean.sum()

    # Feature's name (considering your X a DataFrame)
    feature_names = train.columns

    # Sort to plot in order of importance
    sorted_idx = np.argsort(perm_importance_normalized)[::-1][:15]
    features = np.array(feature_names)[sorted_idx]
    plot_barh(features, np.sort(perm_importance_normalized)[::-1], 'chocolate', name)

    he = svm.predict(test)
    y_test = np.asarray(labeltest)
    for i in range(test.shape[0]):
        print(y_test[i],  he[i])


