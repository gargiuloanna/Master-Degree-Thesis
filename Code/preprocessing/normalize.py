from height_normalization import DimensionlessEquations
import pandas as pd

def normalize(data, mean = True, sides = True):
    equation = DimensionlessEquations()

    for sample in range(len(data)):
        height = data.loc[sample, 'Height (cm)']

        if mean == True:

            #Get features
            stride_length = data.loc[sample, "Stride Length"]
            step_length = data.loc[sample, "Step Length"]

            step_frequency = data.loc[sample, "Step Frequency (Cadence)"]

            single_support =data.loc[sample, "Single Support Time"]
            double_support =data.loc[sample, "Double Support Time"]
            stance_time =data.loc[sample, "Stance Time"]
            swing_time =data.loc[sample, "Swing Time"]
            step_time =data.loc[sample, "Step Time"]
            stride_time =data.loc[sample, "Stride Time"]

            #Normalize features
            normalized_stride = equation.length(stride_length, height)
            normalized_step = equation.length(step_length, height)

            normalized_frequency = equation.gait_cadence(step_frequency, height)

            normalized_single_support = equation.gait_time( single_support,height)
            normalized_double_support =equation.gait_time( double_support,height)
            normalized_stance_time =equation.gait_time(stance_time ,height)
            normalized_swing_time =equation.gait_time(swing_time ,height)
            normalized_step_time =equation.gait_time( step_time,height)
            normalized_stride_time =equation.gait_time( stride_time,height)

            #Insert Normalized Features
            data.loc[sample, 'Normalized Stride Length'] = normalized_stride
            data.loc[sample, 'Normalized Step Length'] = normalized_step

            data.loc[sample, 'Normalized Step Frequency'] = normalized_frequency

            data.loc[sample, 'Normalized Single Support Time'] = normalized_single_support
            data.loc[sample, 'Normalized Double Support Time'] = normalized_double_support
            data.loc[sample, 'Normalized Stance Time'] = normalized_stance_time
            data.loc[sample, 'Normalized Swing Time'] = normalized_swing_time
            data.loc[sample, 'Normalized Step Time'] = normalized_step_time
            data.loc[sample, 'Normalized Stride Time'] = normalized_stride_time

        if sides == True:

            #get features
            left_stride_length =data.loc[sample, "Left Stride Length"]
            right_stride_length =data.loc[sample, "Right Stride Length"]

            left_step_frequency =data.loc[sample, "Left Step Frequency (Cadence)"]
            right_step_frequency = data.loc[sample, "Right Step Frequency (Cadence)"]

            left_step_length = data.loc[sample, "Left Step Length"]
            right_step_length = data.loc[sample, "Right Step Length"]

            left_stance_time = data.loc[sample, "Left Stance Time"]
            right_stance_time = data.loc[sample, "Right Stance Time"]

            left_swing_time = data.loc[sample, "Left Swing Time"]
            right_swing_time = data.loc[sample, "Right Swing Time"]

            left_stride_time = data.loc[sample, "Left Stride Time"]
            right_stride_time = data.loc[sample, "Right Stride Time"]

            single_support = data.loc[sample, "Single Support Time"]
            double_support = data.loc[sample, "Double Support Time"]

            #normalize features
            normalized_left_stride = equation.length(left_stride_length, height)
            normalized_right_stride = equation.length(right_stride_length, height)

            normalized_left_frequency = equation.gait_cadence(left_step_frequency, height)
            normalized_right_frequency = equation.gait_cadence(right_step_frequency, height)

            normalized_left_step = equation.length(left_step_length, height)
            normalized_right_step = equation.length(right_step_length, height)

            normalized_single_support = equation.gait_time(single_support, height)
            normalized_double_support = equation.gait_time(double_support, height)

            normalized_left_stance_time =equation.gait_time(left_stance_time ,height)
            normalized_left_swing_time =equation.gait_time(left_swing_time ,height)
            normalized_left_stride_time =equation.gait_time( left_stride_time,height)

            normalized_right_stance_time =equation.gait_time(right_stance_time ,height)
            normalized_right_swing_time =equation.gait_time(right_swing_time ,height)
            normalized_right_stride_time =equation.gait_time(right_stride_time,height)

            #insert
            data.loc[df.index[sample], 'Normalized Left Stride Length'] = normalized_left_stride
            data.loc[df.index[sample], 'Normalized Right Stride Length'] = normalized_right_stride

            data.loc[sample, 'Normalized Left Step Frequency'] = normalized_left_frequency
            data.loc[sample, 'Normalized Right Step Frequency'] = normalized_right_frequency

            data.loc[df.index[sample], 'Normalized Left Step Length'] = normalized_left_step
            data.loc[df.index[sample], 'Normalized Right Step Length'] = normalized_right_step

            data.loc[sample, 'Normalized Single Support Time'] = normalized_single_support
            data.loc[sample, 'Normalized Double Support Time'] = normalized_double_support
            data.loc[sample, 'Normalized Left Stance Time'] = normalized_left_stance_time
            data.loc[sample, 'Normalized Left Swing Time'] = normalized_left_swing_time
            data.loc[sample, 'Normalized Left Stride Time'] = normalized_left_stride_time

            data.loc[sample, 'Normalized Right Stance Time'] = normalized_right_stance_time
            data.loc[sample, 'Normalized Right Swing Time'] = normalized_right_swing_time
            data.loc[sample, 'Normalized Right Stride Time'] = normalized_right_stride_time

    return data.copy()



if __name__ == '__main__':
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset -features_changed.xlsx')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)


    data = normalize(df, mean=True, sides=True)

    data.to_excel("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-normalized lengths.xlsx")

    data.drop(["Double Support Time","Single Support Time","Right Stride Time","Right Stride Time","Right Swing Time",
                "Left Step Frequency (Cadence)","Right Step Frequency (Cadence)","Left Step Length","Right Step Length",
                "Left Stance Time","Right Stance Time","Left Swing Time","Right Stride Length", "Left Stride Length",
               "Stride Length","Step Length","Step Frequency (Cadence)","Stance Time","Swing Time","Step Time","Stride Time"],
              axis = 1, inplace = True)

    data.to_excel("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-only normalized lengths.xlsx")






