from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from Code.evaluation.predict import predict_score
from Code.preprocessing.preprocess import preprocess
from joblib import dump
import numpy as np

if __name__ == '__main__':

    train, test, labeltrain, labeltest = preprocess()

    seeds = np.random.randint(0, 9999999, size = 100)

    mean_train = []
    mean_oob = []
    mean_test = []

    for seed in seeds:
        name = "RandomforestWholePreprocess "+str(seed)

        #clf = RandomForestClassifier(criterion='gini', max_depth=5, max_features='sqrt', min_samples_split=0.1,min_samples_leaf=1, n_estimators=100, random_state=seed, oob_score=balanced_accuracy_score, class_weight='balanced')

        clf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_split=10,
                                 min_samples_leaf=1, n_estimators=100, random_state=5436398, oob_score=balanced_accuracy_score,
                                 class_weight='balanced')

        clf.fit(train, labeltrain)

        train_score = predict_score(clf, train, labeltrain)
        oob = clf.oob_score_
        test_score = predict_score(clf, test, labeltest)

        mean_train.append(train_score)
        mean_oob.append(oob)
        mean_test.append(test_score)

        print("Train Accuracy: ", train_score)
        print("Training OOB Accuracy ", oob)
        print("Test Accuracy: ", test_score)

        rdf = dump(clf, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/randomforest/" + name + ".joblib")
        print("[RANDOM FOREST] Model Saved")

        print(classification_report(labeltest, clf.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/randomforest/randomforestseeds.txt", "a")
        file.write("\nModel " + name + '\n')
        file.write("Train Accuracy: " + str(predict_score(clf, train, labeltrain)) + '\n')
        file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
        file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')


    file.write("----------------------------------------END RUN----------------------------------------\n")
    file.write("Average on Train: " + str(np.mean(np.array(mean_train))) + "\tSTD: " + str(np.std(np.array(mean_train))) + "\n")
    file.write("Average on OOB: " + str( np.mean( np.array(mean_oob) ) ) + "\tSTD: " + str( np.std( np.array(mean_oob) ) ) +"\n")
    file.write("Average on test: "+ str(np.mean(np.array(mean_test))) + "\tSTD: " + str( np.std( np.array(mean_oob) ) ) + "\n")
    file.write("Seeds " + str(seeds) + '\n')
    file.close()

