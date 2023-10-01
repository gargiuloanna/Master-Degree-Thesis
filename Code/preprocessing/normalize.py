from height_normalization import DimensionlessEquations
import pandas as pd

def normalize(data, mean = True, sides = True):
    equation = DimensionlessEquations()

    for sample in range(len(data)):
        height = data.loc[sample, 'Height (cm)']
        if mean == True and sides == False:

            stride_length = data.loc[sample, "Stride Length"]
            step_length = data.loc[sample, "Step Length"]

        if mean == True and sides == True:

            stride_length = data.loc[sample, "Stride Length"]
            left_stride_length =data.loc[sample, "Left Stride Length"]
            right_stride_length =data.loc[sample, "Right Stride Length"]

            step_length = data.loc[sample, "Step Length"]
            left_step_length = data.loc[sample, "Left Step Length"]
            right_step_length = data.loc[sample, "Right Step Length"]

        if mean == False and sides == True:
            left_stride_length =data.loc[sample, "Left Stride Length"]
            right_stride_length =data.loc[sample, "Right Stride Length"]

            left_step_length = data.loc[sample, "Left Step Length"]
            right_step_length = data.loc[sample, "Right Step Length"]

        if mean == True:
            normalized_stride = equation.length(stride_length, height)
            normalized_step = equation.length(step_length, height)

            data.loc[sample, 'Normalized Stride Length'] = normalized_stride
            data.loc[sample, 'Normalized Step Length'] = normalized_step

        if sides == True:
            normalized_left_stride = equation.length(left_stride_length, height)
            normalized_right_stride =equation.length(right_stride_length, height)

            normalized_left_step =equation.length(left_step_length, height)
            normalized_right_step=equation.length(right_step_length, height)

            data.loc[df.index[sample], 'Normalized Left Stride Length'] = normalized_left_stride
            data.loc[df.index[sample], 'Normalized Right Stride Length'] = normalized_right_stride
            data.loc[df.index[sample], 'Normalized Left Step Length'] = normalized_left_step
            data.loc[df.index[sample], 'Normalized Right Step Length'] = normalized_right_step

    return data.copy()



if __name__ == '__main__':
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Tesi/Data/Dataset.xlsx')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)


    data = normalize(df, mean=True, sides=True)

    data.to_excel("C:/Users/annin/PycharmProjects/Tesi/Data/Dataset-normalized lengths.xlsx")






