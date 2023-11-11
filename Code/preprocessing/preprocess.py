import pandas as pd
from Code.preprocessing.outlier_detection import remove_outliers
from Code.preprocessing.variance_thresholding import variance
from Code.preprocessing.scale import scale
from Code.preprocessing.correlation import correlation
from sklearn.model_selection import train_test_split

d={}
a = {}

#pd
d['pd002']=172
d['pd003']=174
d['pd004']=170
d['pd005']=160
d['pd006']=182
d['pd007']=174
d['pd008']=175

a['pd002']=76
a['pd003']=73
a['pd004']=79
a['pd005']=65
a['pd006']=63
a['pd007']=67
a['pd008']=71
#el
d['el001']=168
d['el002']=173
d['el003']=167
d['el004']=165
d['el006']=172
d['el007']=170
d['el008']=177
d['el009']=173
d['el010']=185

a['el001']=85
a['el002']=96
a['el003']=67
a['el004']=60
a['el006']=61
a['el007']=83
a['el008']=67
a['el009']=78
a['el010']=65
#adults
d['s001']=169
d['s002']=169
d['s003']=173
d['s004']=178
d['s005']=175
d['s006']=176
d['s007']=180
d['s008']=182
d['s009']=182
d['s010']=185
d['s011']=170
d['s012']=168
d['s013']=187

a['s001']=20
a['s002']=34
a['s003']=33
a['s004']=32
a['s005']=50
a['s006']=30
a['s007']=43
a['s008']=27
a['s009']=51
a['s010']=56
a['s011']=30
a['s012']=30
a['s013']=58

def preprocess(file = 'C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/Dataset-onlynormalizedlr.xlsx'):

    df = pd.read_excel(file)
    df.drop(['Unnamed: 0', 'Height (cm)'], axis=1, inplace=True)
    #df.drop(['Height (cm)'], axis=1, inplace=True)
    #Check for outliers and remove them
    df = remove_outliers(df)

    # Feature Correlation
    df = correlation(df)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    # Split data

    train, test, labeltrain, labeltest = train_test_split(data, labels, test_size=0.30, random_state=12345, stratify=labels)
    print(test['Patient'].to_markdown())
    for patient in test['Patient']:
        print(a[patient])
    print(test['Exercise'].to_markdown())
    # Scale data
    train, test = scale(train, test)

    #Remove features with variance <0.01
    train, test = variance(train, test, threshold = 0.4)


    return train, test, labeltrain, labeltest




