import os
import pandas as pd
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

if __name__ == '__main__':

    df_results = pd.DataFrame()
    for root, dirs, files in os.walk("C:\\Users\\annin\PycharmProjects\Master-Degree-Thesis\Code\Data\SmartInsole"):
        for file in files:
            if file.endswith('2tug1.xlsx') or file.endswith('2tug2.xlsx'):
                print("\nfile ", root, file)
                df = pd.read_excel(os.path.join(root, file))

                values = df.value_counts('Activity- Label level 1')
                length = df.shape[0]
                time_brute = float(float(length) /100)
                vals = values['WAL']
                motions = length - vals
                time_formula = float((float(vals/3) + motions) /100)
                print("WALKING MOTION: ", vals, "OTHER" , motions)
                pos = len(df_results)
                df_results.loc[pos, 'Patient'] = (file.split('_')[0])
                df_results.loc[pos, 'Height'] = d[file.split('_')[0]]
                df_results.loc[pos, 'Age'] = a[file.split('_')[0]]
                df_results.loc[pos,'TUG Time'] = (time_brute)
                df_results.loc[pos,'TUG Time / 3.33333'] = (float(time_brute) / 3.33333333333)
                df_results.loc[pos,'TUG Time Formula'] = (time_formula)
                for name, val in values.items():
                    df_results.loc[pos, name]=val/100
                #df_results = pd.concat([df_results, pd.Series(d)], ignore_index=True)
                print(df_results)

    df_results.to_excel("C:\\Users\\annin\PycharmProjects\Master-Degree-Thesis\Code\Data\TUG_TIMES.xlsx")


