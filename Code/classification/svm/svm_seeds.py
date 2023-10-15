from sklearn.svm import SVC
from sklearn.metrics import classification_report
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
        name = "SVM_"+str(seed)

        #clf = RandomForestClassifier(criterion='gini', max_depth=5, max_features='sqrt', min_samples_split=0.1,min_samples_leaf=1, n_estimators=100, random_state=seed, oob_score=balanced_accuracy_score, class_weight='balanced')

        svm = SVC(random_state=0, class_weight='balanced', kernel = 'poly',
                  C = 0.1, degree = 1, coef0 = 1, decision_function_shape='ovo')


        svm.fit(train, labeltrain)

        train_score = predict_score(svm, train, labeltrain)
        test_score = predict_score(svm, test, labeltest)

        mean_train.append(train_score)
        mean_test.append(test_score)

        print("Train Accuracy: ", train_score)
        print("Test Accuracy: ", test_score)

        rdf = dump(svm, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/svm/" + name + ".joblib")
        print("[RANDOM FOREST] Model Saved")

        print(classification_report(labeltest, svm.predict(test),  target_names=['Elderly', 'Parkinson', 'Adults'],  digits=3, output_dict=False, zero_division='warn'))

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/svm/svmseeds.txt", "a")
        file.write("\nModel " + name + '\n')
        file.write("Train Accuracy: " + str(predict_score(svm, train, labeltrain)) + '\n')
        file.write("Test Accuracy: " + str(predict_score(svm, test, labeltest)) + '\n')


    file.write("----------------------------------------END RUN----------------------------------------\n")
    file.write("Average on Train: " + str(np.mean(np.array(mean_train))) + "\tSTD: " + str(np.std(np.array(mean_train))) + "\n")
    file.write("Average on OOB: " + str( np.mean( np.array(mean_oob) ) ) + "\tSTD: " + str( np.std( np.array(mean_oob) ) ) +"\n")
    file.write("Average on test: "+ str(np.mean(np.array(mean_test))) + "\tSTD: " + str( np.std( np.array(mean_oob) ) ) + "\n")
    file.write("Seeds " + str(seeds) + '\n')
    file.close()