import pandas as pd
from Code.plotting.plots import plot_correlation




if __name__ == '__main__':

    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/data_varianced/Dataset-only normalized lengths/Dataset-only normalized lengths_0.99.xlsx')
    df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis=1, inplace=True)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    #first_batch =data.iloc[:, :int(len(data.columns)/2)]
    #second_batch = data.iloc[:, int(len(data.columns)/2):]
    #plot_correlation(first_batch, name = 'first')
    #plot_correlation(second_batch, name='second')
    plot_correlation(df, name = 'corr_label')