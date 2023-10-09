from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from Code.evaluation.predict import predict_score
from joblib import dump
from Code.plotting.plots import confusion_matrix, random_forest_fimp, permutation_imp, feature_importance, plot_correlation
from Code.preprocessing.scale import scale
from  sklearn.metrics import balanced_accuracy_score
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #['Dataset -features_changed.xlsx','Dataset -features_changed - lr.xlsx','Dataset -features_changed - mean.xlsx' ,'Dataset-normalized lengths.xlsx', 'Dataset-normalized lengths - lr.xlsx','Dataset-normalized lengths - mean.xlsx','Dataset-only normalized lengths.xlsx','Dataset-only normalized lengths -lr.xlsx','Dataset-only normalized lengths - mean.xlsx']
    for root, dirs, files in os.walk('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/Dataset-only normalized lengths/'):
        for file in files:
            df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/Dataset-only normalized lengths/' + file)
            df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

            data = df.iloc[:, :-1]
            labels = df.iloc[:, -1]
            
            train, test, labeltrain, labeltest =train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
            train, test = scale(train, test)

            name = "Randomforest " + file
            clf = RandomForestClassifier(criterion = 'entropy', max_depth = 8, max_features= 'sqrt', min_samples_split=10,
                                         min_samples_leaf=1, n_estimators=100,random_state=0, oob_score=balanced_accuracy_score,
                                         class_weight='balanced')
            clf.fit(train, labeltrain)
            print("Training OOB Accuracy ", clf.oob_score_)
            rdf = dump(clf, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/models/" + name + "WITHVARIANCE.joblib")
            print("[RANDOM FOREST] Model Saved")

            print("Test Accuracy: ", predict_score(clf, test, labeltest))
            file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/classification/performances.txt", "a")
            file.write("Model " +name + '\n')
            file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
            file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')
            file.close()

            # plots
            if not os.path.exists("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + name):
                os.mkdir("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Plots/" + name)
            confusion_matrix(clf, test, labeltest, name = name)
            plt.close()
            feature_importance(clf, data.columns,name = name)
            plt.close()
            random_forest_fimp(clf, data.columns, name = name)
            plt.close()
            permutation_imp(clf, data, labels, name = name)
            plt.close()


