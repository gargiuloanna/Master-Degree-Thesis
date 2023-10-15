from sklearn.ensemble import RandomForestClassifier
from Code.evaluation.predict import predict_score
from joblib import dump
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance
from  sklearn.metrics import balanced_accuracy_score
import numpy as np
from Code.preprocessing.preprocess import preprocess
from sklearn.metrics import classification_report


if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()

    seed = 5436398
    name = "RandomforestWholePreprocess "+str(seed)

    clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=5436398, oob_score=balanced_accuracy_score,
                                 class_weight='balanced')

    clf.fit(train, labeltrain)

    # Performance
    print("Train Accuracy: ", predict_score(clf, train, labeltrain))
    print("Training OOB Accuracy ", clf.oob_score_)
    print("Test Accuracy: ", predict_score(clf, test, labeltest))

    #Save Model
    rdf = dump(clf, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/models/" + name + ".joblib")
    print("[RANDOM FOREST] Model Saved")

    print(classification_report(labeltest, clf.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

    # Write to File
    file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/classification/randomforest.txt", "a")
    file.write("\nModel " + name + '\n')
    file.write("Train Accuracy: " + str(predict_score(clf, train, labeltrain)) + '\n')
    file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
    file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')
    file.close()

    # Plots
    confusion_matrix(clf, test, labeltest, name=name)
    feature_importance(clf, train.columns, name=name)
    random_forest_fimp(clf, train.columns, name=name)
    permutation_imp(clf, test, labeltest, name=name)

    #Misclassified Samples
    y_test = np.asarray(labeltest)
    misclassified = np.where(y_test != clf.predict(test))
    for ind in misclassified[0]:
        print(test.loc[ind])


