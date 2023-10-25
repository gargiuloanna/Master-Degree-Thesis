from kfda_class import Kfda
from Code.preprocessing.preprocess import preprocess
from sklearn.feature_selection import SequentialFeatureSelector

if __name__ == '__main__':
        i = 0.5
        name= "kfda/KFDA_SFS"
        train, test, labeltrain, labeltest = preprocess()

        file = open("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/results/logs/kfda/kfda.txt", "a")

        cls = Kfda(n_components=2, kernel='poly')
        sfs = SequentialFeatureSelector(cls, n_features_to_select=i, tol=None, direction='forward', scoring='balanced_accuracy', cv=10,
                                               n_jobs=-1)
        print(sfs.fit(train, labeltrain))

        cls.fit(sfs.transform(train), labeltrain)
        file.write("Train score: " + str(cls.score(train, labeltrain)) + '\n')
        file.write("Test score: " + str(cls.score(test, labeltest)) + '\n')

        print(sfs.get_feature_names_out(train.columns))
        print(cls.score(sfs.transform(test), labeltest))

                # Write to File
        file.write("\nModel " + name + str(i) + '\n')
        file.write("Train Accuracy: " + str(cls.score(sfs.transform(train), labeltrain)) + '\n')
        file.write("Test Accuracy: " + str(cls.score(sfs.transform(test), labeltest)) + '\n')
        file.write("Features: " + str(sfs.get_feature_names_out(train.columns)) + '\n')
        file.close()
