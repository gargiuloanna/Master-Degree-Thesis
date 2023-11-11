from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,balanced_accuracy_score
from Code.evaluation.predict import predict_score
from Code.plotting.plots import permutation_imp, feature_importance
from Code.preprocessing.preprocess import preprocess
import numpy as np
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


    train, test, labeltrain, labeltest = preprocess()
    seed = 5940440
    name = "RandomForest_last_"+str(seed)

    clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=seed, oob_score=balanced_accuracy_score,
                                 class_weight='balanced')

    clf.fit(train, labeltrain)

    # Performance
    print("Train Accuracy: ", predict_score(clf, train, labeltrain))
    print("Training OOB Accuracy ", clf.oob_score_)
    print("Test Accuracy: ", predict_score(clf, test, labeltest))

    print(classification_report(labeltest, clf.predict(test),  digits=3, output_dict=False, zero_division='warn'))

    # Write to File

    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/randomforest/randomforest.txt", "a")
    file.write("\nModel " + name + '\n')
    file.write("Train Accuracy: " + str(predict_score(clf, train, labeltrain)) + '\n')
    file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
    file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')
    file.close()

    # Plots
    confusion_matrix(clf, test, labeltest, 'Blues', name=name)
    feature_importance(clf, train.columns, 'skyblue', name=name)
    permutation_imp(clf, test, labeltest, 'skyblue', name=name)

    #Misclassified Samples
    he = clf.predict(test)
    y_test = np.asarray(labeltest)
    for i in range(test.shape[0]):
        print(y_test[i],  he[i])

