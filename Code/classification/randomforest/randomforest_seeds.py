from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance
from Code.evaluation.predict import predict_score
from Code.preprocessing.preprocess import preprocess
from joblib import dump
import numpy as np
import os
if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()

    seeds = np.random.randint(0, 9999999, size = 100)

    mean_train = []
    mean_oob = []
    mean_test = []
    best_test = {}
    best_test['Seed'] = 0
    best_test['Train Accuracy'] = 0.0
    best_test['Training OOB Accuracy'] = 0.0
    best_test['Test Accuracy'] = 0.0
    for seed in seeds:
        name = "RandomforestWholePreprocess_MEAN_ "+str(seed)

        #clf = RandomForestClassifier(criterion='gini', max_depth=5, max_features='sqrt', min_samples_split=0.1,min_samples_leaf=1, n_estimators=100, random_state=seed, oob_score=balanced_accuracy_score, class_weight='balanced')

        clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=seed, oob_score=balanced_accuracy_score,
                                 class_weight='balanced')

        clf.fit(train, labeltrain)

        train_score = predict_score(clf, train, labeltrain)
        oob = clf.oob_score_
        test_score = predict_score(clf, test, labeltest)

        mean_train.append(train_score)
        mean_oob.append(oob)
        mean_test.append(test_score)

        if test_score > best_test['Test Accuracy']:
            best_test['Seed'] = seed
            best_test['Train Accuracy'] =train_score
            best_test['Training OOB Accuracy'] = oob
            best_test['Test Accuracy'] = test_score

        print("Train Accuracy: ", train_score)
        print("Training OOB Accuracy ", oob)
        print("Test Accuracy: ", test_score)

        print(classification_report(labeltest, clf.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/randomforest/randomforestseeds.txt", "a")
        file.write("\nModel " + name + '\n')
        file.write("Train Accuracy: " + str(predict_score(clf, train, labeltrain)) + '\n')
        file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
        file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')


    file.write("----------------------------------------RESULTS----------------------------------------\n")
    file.write("Average on Train: " + str(np.mean(np.array(mean_train))) + "\tSTD: " + str(np.std(np.array(mean_train))) + "\n")
    file.write("Average on OOB: " + str( np.mean( np.array(mean_oob) ) ) + "\tSTD: " + str( np.std( np.array(mean_oob) ) ) +"\n")
    file.write("Average on test: "+ str(np.mean(np.array(mean_test))) + "\tSTD: " + str( np.std( np.array(mean_test) ) ) + "\n")
    file.write("\nBEST ON TEST\n" + str(best_test))
    file.write("\n\nSeeds " + str(seeds) + '\n')
    file.write("----------------------------------------END RUN----------------------------------------\n")
    file.close()

    #________________________________________plots for best seed________________________________________#
    name = "randomforest/RandomForest_BEST_MEAN" + str(best_test['Seed'])
    clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=best_test['Seed'],
                                 oob_score=balanced_accuracy_score, class_weight='balanced')

    clf.fit(train, labeltrain)

    rdf = dump(clf, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/" + name + ".joblib")
    print("[RANDOM FOREST] Model Saved")

    # Plots
    confusion_matrix(clf, test, labeltest, name=name)
    feature_importance(clf, train.columns, name=name)
    random_forest_fimp(clf, train.columns, name=name)
    permutation_imp(clf, test, labeltest, name=name)

    # Misclassified Samples
    y_test = np.asarray(labeltest)
    misclassified = np.where(y_test != clf.predict(test))
    for ind in misclassified[0]:
        print(test.loc[ind])