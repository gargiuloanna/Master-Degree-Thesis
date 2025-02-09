from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from  sklearn.metrics import balanced_accuracy_score
from Code.evaluation.predict import predict_score
from Code.preprocessing.preprocess import preprocess
from joblib import dump

if __name__ == '__main__':

        train, test, labeltrain, labeltest, _ =preprocess()

        name="RandomForest_GridSearch_gait_patient"

        print("[RANDOM FOREST] Searching best params with GridSearchCV")

        rdf_model = RandomForestClassifier(random_state=0,
                                           oob_score=balanced_accuracy_score,
                                           class_weight='balanced')
        param_grid = {
            'n_estimators': [70, 100, 110, 120, 130, 140, 150],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [0.1, 0.5, 1, 5, 10, 20],
            'max_features': ['sqrt', 'log2']
        }

        rdf_gridcv = GridSearchCV(rdf_model, param_grid=param_grid, cv=10, scoring='balanced_accuracy',  error_score=0, n_jobs=-1, verbose=3, refit=True)
        rdf_gridcv.fit(train, labeltrain)

        print(f"[RANDOM FOREST] Best random forest with params: {rdf_gridcv.best_params_} and score {rdf_gridcv.best_score_}")
        # save model
        clf = RandomForestClassifier(n_estimators= rdf_gridcv.best_params_['n_estimators'], criterion = rdf_gridcv.best_params_['criterion'] ,max_depth= rdf_gridcv.best_params_['max_depth'],
                                     min_samples_split= rdf_gridcv.best_params_['min_samples_split'], min_samples_leaf= rdf_gridcv.best_params_['min_samples_leaf'], max_features=rdf_gridcv.best_params_['max_features'],
                                     random_state = 0, oob_score = balanced_accuracy_score, class_weight='balanced')

        clf.fit(train, labeltrain)
        print("Training OOB Accuracy ", clf.oob_score_)
        print("Test Accuracy: ", predict_score(rdf_gridcv.best_estimator_, test, labeltest))

        rdf = dump(rdf_gridcv.best_estimator_, "C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/models/randomforest/" + name + ".joblib")
        print("[RANDOM FOREST] Model Saved")

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/randomforest/randomforestgridsearch.txt", "a")
        file.write(f"[RANDOM FOREST] Best random forest with params: {rdf_gridcv.best_params_} and score {rdf_gridcv.best_score_}")
        file.write("\nModel " + name + '\n')
        file.write("Train OOB Accuracy: " + str(clf.oob_score_) + '\n')
        file.write("Test Accuracy: " + str(predict_score(clf, test, labeltest)) + '\n')
        file.close()


