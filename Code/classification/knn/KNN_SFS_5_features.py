from sklearn.neighbors import KNeighborsClassifier
from Code.preprocessing.preprocess import preprocess
if __name__ == '__main__':
        f = ['Height (cm)' ,'Left Stride Velocity','Right Stride Velocity',
         'Left Gait Variability','Left Stance Phase', 'Right Stance Phase',
         'Left Swing Phase' ,'Right Swing Phase' ,'Single Support Phase',
         'Double Support Phase', 'Left Heel-Toe Ratio' 'Right Heel-Toe Ratio',
         'Left - Right Single Ratio', 'Heel Ratio' 'Toe Ratio',
         'Avg Left Heel Pressure Skewness' ,'Avg Left Toe Pressure Skewness',
         'Left Acceleration X Skewness' ,'Left Acceleration Y Skewness',
         'Left Acceleration Z Skewness', 'Left Angular X Skewness',
         'Left Angular Y Skewness', 'Left Angular Z Skewness',
         'Left Total Force Skewness' ,'Left COP X Skewness' 'Left COP Y Skewness',
         'Avg Right Heel Pressure Skewness' ,'Avg Right Toe Pressure Skewness',
         'Right Acceleration X Skewness', 'Right Acceleration Y Skewness',
         'Right Acceleration Z Skewness' ,'Right Angular X Skewness',
         'Right Angular Y Skewness', 'Right Angular Z Skewness',
         'Right Total Force Skewness' ,'Right COP X Skewness',
         'Right COP Y Skewness', 'Avg Left Heel Pressure Kurtosis',
         'Avg Left Toe Pressure Kurtosis' ,'Left Acceleration X Kurtosis',
         'Left Angular X Kurtosis', 'Left Angular Y Kurtosis',
         'Left Angular Z Kurtosis' ,'Left Total Force Kurtosis',
         'Left COP X Kurtosis', 'Left COP Y Kurtosis',
         'Avg Right Heel Pressure Kurtosis' ,'Right Acceleration X Kurtosis',
         'Right Acceleration Y Kurtosis', 'Right Acceleration Z Kurtosis',
         'Right Angular X Kurtosis', 'Right Angular Y Kurtosis',
         'Right Total Force Kurtosis' ,'Right COP Y Kurtosis']

        name = "knn/KNN_SFS"
        train, test, labeltrain, labeltest = preprocess()
        clf = KNeighborsClassifier(algorithm='auto',  metric='minkowski', n_neighbors=2, p=1, weights='distance', n_jobs=-1)
        columns_to_remove = set(train.columns).difference(set(f))

        train.drop(columns_to_remove, axis = 1, inplace = True)
        test.drop(columns_to_remove, axis = 1, inplace = True)
        print(test.to_markdown())
        clf.fit(train, labeltrain)
        print(clf.score(test, labeltest))

        #'Height (cm)'   -0.73
        #'Left Acceleration X Skewness'  -0.8
        # 'Right Acceleration Z Kurtosis' -0.86
        # 'Right COP Y Skenwss' -0.93
        # 'Avg Left Toe Pressure Skewness' -0.95
