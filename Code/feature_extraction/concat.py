import pandas as pd
import os

def concat():
    frames = []
    for root, dirs, files in os.walk('/Code/Data/features_changed/'):
        for file in files:
            print(file)
            frames.append(pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/features_changed/' + file))

    result = pd.concat(frames, ignore_index=True)
    result.drop('Unnamed: 0', axis=1, inplace=True)
    result.to_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset -features_changed.xlsx')
    return result

def mean_values():
    result = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset.xlsx')
    result.drop('Unnamed: 0', axis=1, inplace=True)

    el_norm = [result['Patient'][i].startswith('el') and result['Exercise'][i].startswith('1norm') for i in range(0, len(result))]
    s_norm =[result['Patient'][i].startswith('s') and result['Exercise'][i].startswith('1norm') for i in range(0, len(result))]
    p_norm = [result['Patient'][i].startswith('pd') and result['Exercise'][i].startswith('1norm') for i in range(0, len(result))]

    el_high = [result['Patient'][i].startswith('el_norm') and result['Exercise'][i].startswith('1high') for i in range(0, len(result))]
    s_high= [result['Patient'][i].startswith('s') and result['Exercise'][i].startswith('1high') for i in range(0, len(result))]
    p_high = [result['Patient'][i].startswith('pd') and result['Exercise'][i].startswith('1high') for i in range(0, len(result))]

    el_slow = [result['Patient'][i].startswith('el_norm') and result['Exercise'][i].startswith('1slow') for i in range(0, len(result))]
    s_slow = [result['Patient'][i].startswith('s') and result['Exercise'][i].startswith('1slow') for i in range(0, len(result))]
    p_slow = [result['Patient'][i].startswith('pd') and result['Exercise'][i].startswith('1slow') for i in range(0, len(result))]

    el_tug = [result['Patient'][i].startswith('el_norm') and result['Exercise'][i].startswith('2tug') for i in range(0, len(result))]
    s_tug = [result['Patient'][i].startswith('s') and result['Exercise'][i].startswith('2tug') for i in range(0, len(result))]
    p_tug = [result['Patient'][i].startswith('pd') and result['Exercise'][i].startswith('2tug') for i in range(0, len(result))]

    mean_el_norm = result.loc[el_norm].mean(numeric_only = True)
    mean_s_norm = result.loc[s_norm].mean(numeric_only=True)
    mean_pd_norm = result.loc[p_norm].mean(numeric_only=True)

    mean_el_high = result.loc[el_high].mean(numeric_only=True)
    mean_s_high = result.loc[s_high].mean(numeric_only=True)
    mean_pd_high = result.loc[p_high].mean(numeric_only=True)

    mean_el_slow = result.loc[el_slow].mean(numeric_only=True)
    mean_s_slow = result.loc[s_slow].mean(numeric_only=True)
    mean_pd_slow = result.loc[p_slow].mean(numeric_only=True)

    mean_el_tug = result.loc[el_tug].mean(numeric_only=True)
    mean_s_tug = result.loc[s_tug].mean(numeric_only=True)
    mean_pd_tug = result.loc[p_tug].mean(numeric_only=True)

    print("EL NORM\n", mean_el_norm.to_markdown())
    print("\nS NORM\n", mean_s_norm.to_markdown())
    print("\nPD NORM\n", mean_pd_norm.to_markdown())

    print("\nEL HIGH\n", mean_el_high.to_markdown())
    print("\nS HIGH\n", mean_s_high.to_markdown())
    print("\nPD HIGH\n", mean_pd_high.to_markdown())

    print("\nEL SLOW\n", mean_el_slow.to_markdown())
    print("\nS SLOW\n", mean_s_slow.to_markdown())
    print("\nPD SLOW\n", mean_pd_slow.to_markdown())

    print("\nEL TUG\n", mean_el_tug.to_markdown())
    print("\nS TUG\n", mean_s_tug.to_markdown())
    print("\nPD TUG\n", mean_pd_tug.to_markdown())



if __name__ == '__main__':

    concat()
    #mean_values()

