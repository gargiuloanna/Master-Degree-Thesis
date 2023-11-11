# -*- coding: utf-8 -*-
import pandas as pd
from Code.feature_extraction.feature_functions.utilities import get_hes_toes, check_length_difference
from Code.feature_extraction.feature_functions.stride import compute_stride_length, compute_stride_velocity, compute_stride_time, compute_gait_variability
from Code.feature_extraction.feature_functions.step import compute_step_frequency, compute_step_length, compute_step_time
from Code.feature_extraction.feature_functions.stance import compute_stance_time, compute_stance_phase
from Code.feature_extraction.feature_functions.single_support import compute_single_support_time, compute_single_support_phase
from Code.feature_extraction.feature_functions.double_support import compute_double_support_time, compute_double_support_phase
from Code.feature_extraction.feature_functions.swing import compute_swing_phase, compute_swing_time
from Code.feature_extraction.feature_functions.walk_ratio import compute_walk_ratio
from Code.feature_extraction.feature_functions.R_Index import single_pressure_ratio, double_pressure_ratio
from Code.feature_extraction.feature_functions.gait_speed import compute_gait_speed
from Code.feature_extraction.feature_functions.approximate_entropy import approximate_entropy

from scipy.stats import kurtosis
from scipy.stats import skew

def features(dn_complete, name):
    # geet heel strikes and toe offs
    left_heel_strike, left_toe_off = get_hes_toes(dn_complete)
    right_heel_strike, right_toe_off = get_hes_toes(dn_complete, side='Right')

    # fix windows
    left_windows_heel, left_windows_toe = check_length_difference(left_heel_strike, left_toe_off)
    right_windows_heel, right_windows_toe = check_length_difference(right_heel_strike, right_toe_off)

    # Stride Time, Length, Velocity
    gait_cycle, left_stride_time, right_stride_time = compute_stride_time(dn_complete, left_windows_heel, right_windows_heel)
    stride_length, left_stride_length, right_stride_length = compute_stride_length(left_windows_heel, right_windows_heel)
    gait_velocity, left_gait_velocity, right_gait_velocity = compute_stride_velocity(left_stride_length, right_stride_length, left_stride_time, right_stride_time)
    # Gait Variability
    gait_variability, left_gait_variability, right_gait_variability = compute_gait_variability(dn_complete, left_windows_heel, right_windows_heel)

    # Step Time, Length, Frequency
    step_time = compute_step_time(dn_complete, left_windows_heel, right_windows_heel)
    step_length, left_step_length, right_step_length = compute_step_length(left_windows_heel, right_windows_heel)
    #step_length = compute_step_length(left_windows_heel, right_windows_heel)
    step_frequency, left_step_frequency, right_step_frequency = compute_step_frequency(dn_complete, left_windows_heel, right_windows_heel)

    # Stance Time, Phase
    stance_time, left_stance_time, right_stance_time = compute_stance_time(dn_complete, left_windows_heel, right_windows_heel, left_windows_toe, right_windows_toe)
    stance_phase = compute_stance_phase(stance_time, gait_cycle)
    left_stance_phase = compute_stance_phase(left_stance_time, left_stride_time)
    right_stance_phase = compute_stance_phase(right_stance_time, right_stride_time)

    # Single Support
    single_support_time = compute_single_support_time(dn_complete, left_windows_toe, right_windows_toe, left_windows_heel, right_windows_heel)
    single_support_phase = compute_single_support_phase(single_support_time, gait_cycle)

    # Double Support
    db_support_time = compute_double_support_time(dn_complete, left_windows_heel, right_windows_heel, left_windows_toe, right_windows_toe)
    db_support_phase = compute_double_support_phase(db_support_time, gait_cycle)

    # Swing Time & Phase
    swing_time, swing_time_left, swing_time_right = compute_swing_time(dn_complete, left_windows_toe, right_windows_toe, left_windows_heel, right_windows_heel)
    swing_phase = compute_swing_phase(swing_time, gait_cycle)
    left_swing_phase = compute_stance_phase(swing_time_left, left_stride_time)
    right_swing_phase = compute_stance_phase(swing_time_right, right_stride_time)

    # Walk Ratio
    walk_ratio, left_walk_ratio, right_walk_ratio = compute_walk_ratio(left_step_length, left_step_frequency, right_step_length, right_step_frequency)

    # Gait Speed
    gait_speed = compute_gait_speed(dn_complete)

    # Single Ratio Index
    left_ratio = single_pressure_ratio(dn_complete, side='left')
    right_ratio = single_pressure_ratio(dn_complete, side='right')

    # Double Ratio Index
    single_ratio, double_toe_ratio, double_heel_ratio = double_pressure_ratio(dn_complete)

    # Skewness & Kurtosis
    '''
    # Left Heel and Toe
    left_heel_skew = skew(dn_complete['avg left heel pressure'])
    left_heel_kurtosis = kurtosis(dn_complete['avg left heel pressure'])
    left_toe_skew = skew(dn_complete['avg left toe pressure'])
    left_toe_kurtosis = kurtosis(dn_complete['avg left toe pressure'])

    # Left Accelerations
    left_accx_skew = skew(dn_complete['left acceleration X[g]'])
    left_accx_kurtosis = kurtosis(dn_complete['left acceleration X[g]'])
    left_accy_skew = skew(dn_complete['left acceleration Y[g]'])
    left_accy_kurtosis = kurtosis(dn_complete['left acceleration Y[g]'])
    left_accz_skew = skew(dn_complete['left acceleration Z[g]'])
    left_accz_kurtosis = kurtosis(dn_complete['left acceleration Z[g]'])

    # left Angular
    left_angx_skew = skew(dn_complete['left angular X[dps]'])
    left_angx_kurtosis = kurtosis(dn_complete['left angular X[dps]'])
    left_angy_skew = skew(dn_complete['left angular Y[dps]'])
    left_angy_kurtosis = kurtosis(dn_complete['left angular Y[dps]'])
    left_angz_skew = skew(dn_complete['left angular Z[dps]'])
    left_angz_kurtosis = kurtosis(dn_complete['left angular Z[dps]'])

    # left total force
    left_force_skew = skew(dn_complete['left total force[N]'])
    left_force_kurtosis = kurtosis(dn_complete['left total force[N]'])

    # left center of pressure
    left_cntx_skew = skew(dn_complete['left center of pressure X[%]'])
    left_cntx_kurtosis = kurtosis(dn_complete['left center of pressure X[%]'])
    left_cnty_skew = skew(dn_complete['left center of pressure Y[%]'])
    left_cnty_kurtosis = kurtosis(dn_complete['left center of pressure Y[%]'])

    # Right Heel and Toe
    right_heel_skew = skew(dn_complete['avg right heel pressure'])
    right_heel_kurtosis = kurtosis(dn_complete['avg right heel pressure'])
    right_toe_skew = skew(dn_complete['avg right toe pressure'])
    right_toe_kurtosis = kurtosis(dn_complete['avg right toe pressure'])

    # Right Accelerations
    right_accx_skew = skew(dn_complete['right acceleration X[g]'])
    right_accx_kurtosis = kurtosis(dn_complete['right acceleration X[g]'])
    right_accy_skew = skew(dn_complete['right acceleration Y[g]'])
    right_accy_kurtosis = kurtosis(dn_complete['right acceleration Y[g]'])
    right_accz_skew = skew(dn_complete['right acceleration Z[g]'])
    right_accz_kurtosis = kurtosis(dn_complete['right acceleration Z[g]'])

    # Right Angular
    right_angx_skew = skew(dn_complete['right angular X[dps]'])
    right_angx_kurtosis = kurtosis(dn_complete['right angular X[dps]'])
    right_angy_skew = skew(dn_complete['right angular Y[dps]'])
    right_angy_kurtosis = kurtosis(dn_complete['right angular Y[dps]'])
    right_angz_skew = skew(dn_complete['right angular Z[dps]'])
    right_angz_kurtosis = kurtosis(dn_complete['right angular Z[dps]'])

    # right total force
    right_force_skew = skew(dn_complete['right total force[N]'])
    right_force_kurtosis = kurtosis(dn_complete['right total force[N]'])

    # right center of pressure
    right_cntx_skew = skew(dn_complete['right center of pressure X[%]'])
    right_cntx_kurtosis = kurtosis(dn_complete['right center of pressure X[%]'])
    right_cnty_skew = skew(dn_complete['right center of pressure Y[%]'])
    right_cnty_kurtosis = kurtosis(dn_complete['right center of pressure Y[%]'])
    # Approximate Entropy
    left_steps_number = len(get_windows(left_heel_strike))
    right_steps_number = len(get_windows(right_heel_strike))
    step_number = int((right_steps_number + left_steps_number) / 2)
    # Heel and Toe
    print("Approximate Entropy heel and toe")
    left_heel_ae = approximate_entropy(dn_complete['avg left heel pressure'], step_number)
    right_heel_ae = approximate_entropy(dn_complete['avg right heel pressure'], step_number)

    left_toe_ae = approximate_entropy(dn_complete['avg left toe pressure'], step_number)
    right_toe_ae = approximate_entropy(dn_complete['avg right toe pressure'], step_number)

    # Accelerations
    print("Approximate Entropy accelerations")
    left_accx_ae = approximate_entropy(dn_complete['left acceleration X[g]'], step_number)
    right_accx_ae = approximate_entropy(dn_complete['right acceleration X[g]'], step_number)

    left_accy_ae = approximate_entropy(dn_complete['left acceleration Y[g]'], step_number)
    right_accy_ae = approximate_entropy(dn_complete['right acceleration Y[g]'], step_number)

    left_accz_ae = approximate_entropy(dn_complete['left acceleration Z[g]'], step_number)
    right_accz_ae = approximate_entropy(dn_complete['right acceleration Z[g]'], step_number)

    # Angular
    print("Approximate Entropy angular acc")
    left_angx_ae = approximate_entropy(dn_complete['left angular X[dps]'], step_number)
    right_angx_ae = approximate_entropy(dn_complete['right angular X[dps]'], step_number)

    left_angy_ae = approximate_entropy(dn_complete['left angular Y[dps]'], step_number)
    right_angy_ae = approximate_entropy(dn_complete['right angular Y[dps]'], step_number)

    left_angz_ae = approximate_entropy(dn_complete['left angular Z[dps]'], step_number)
    right_angz_ae = approximate_entropy(dn_complete['right angular Z[dps]'], step_number)

    # total force
    print("Approximate Entropy total force")
    left_force_ae = approximate_entropy(dn_complete['left total force[N]'], step_number)
    right_force_ae = approximate_entropy(dn_complete['right total force[N]'], step_number)

    # center of pressure
    print("Approximate cop")
    left_cntx_ae = approximate_entropy(dn_complete['left center of pressure X[%]'], step_number)
    right_cntx_ae = approximate_entropy(dn_complete['right center of pressure X[%]'], step_number)

    left_cnty_ae = approximate_entropy(dn_complete['left center of pressure Y[%]'], step_number)
    right_cnty_ae = approximate_entropy(dn_complete['right center of pressure Y[%]'], step_number)
    '''
    # Feature insert

    features = pd.DataFrame()
    old_features = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/fixed/' +  name)

    features.loc[0, 'Patient'] = patient
    features.loc[0, 'Exercise'] = task

    # Stride_Time (3)
    features.loc[0, 'Stride Time'] = gait_cycle
    features.loc[0, 'Left Stride Time'] = left_stride_time
    features.loc[0, 'Right Stride Time'] = right_stride_time

    # Stride Length (3)
    features.loc[0, 'Stride Length'] = stride_length
    features.loc[0, 'Left Stride Length'] = left_stride_length
    features.loc[0, 'Right Stride Length'] = right_stride_length

    # Gait Variability
    features.loc[0, 'Gait Variability'] = gait_variability
    features.loc[0, 'Left Gait Variability'] = left_gait_variability
    features.loc[0, 'Right Gait Variability'] = right_gait_variability

    # Step Time
    features.loc[0, 'Step Time'] = step_time

    # Step Length (3)
    features.loc[0, 'Step Length'] = step_length
    features.loc[0, 'Left Step Length'] = left_step_length
    features.loc[0, 'Right Step Length'] = right_step_length

    # Step Frequency (3)
    features.loc[0, 'Step Frequency (Cadence)'] = step_frequency
    features.loc[0, 'Left Step Frequency (Cadence)'] = left_step_frequency
    features.loc[0, 'Right Step Frequency (Cadence)'] = right_step_frequency

    # Stance Time (3)
    features.loc[0, 'Stance Time'] = stance_time
    features.loc[0, 'Left Stance Time'] = left_stance_time
    features.loc[0, 'Right Stance Time'] = right_stance_time

    # Stance Phase
    features.loc[0, 'Stance Phase'] = stance_phase
    features.loc[0, 'Left Stance Phase'] = left_stance_phase
    features.loc[0, 'Right Stance Phase'] = right_stance_phase

    # Single Support Time
    features.loc[0, 'Single Support Time'] = single_support_time

    # Single Support Phase
    features.loc[0, 'Single Support Phase'] = single_support_phase

    # Double Support Time
    features.loc[0, 'Double Support Time'] = db_support_time

    # Double Support Phase
    features.loc[0, 'Double Support Phase'] = db_support_phase

    # Swing Time (3)
    features.loc[0, 'Swing Time'] = swing_time
    features.loc[0, 'Left Swing Time'] = swing_time_left
    features.loc[0, 'Right Swing Time'] = swing_time_right

    # Swing Phase
    features.loc[0, 'Swing Phase'] = swing_phase
    features.loc[0, 'Left Swing Phase'] = left_swing_phase
    features.loc[0, 'Right Swing Phase'] = right_swing_phase
    # Stride Velocity (3)
    features.loc[0, 'Stride Velocity'] = gait_velocity
    features.loc[0, 'Left Stride Velocity'] = left_gait_velocity
    features.loc[0, 'Right Stride Velocity'] = right_gait_velocity

    # Walk Ratio (3)
    features.loc[0, 'Walk Ratio'] = walk_ratio
    features.loc[0, 'Left Walk Ratio'] = left_walk_ratio
    features.loc[0, 'Right Walk Ratio'] = right_walk_ratio

    # Gait Speed
    features.loc[0, 'Gait Speed'] = gait_speed

    # Ratio
    features.loc[0, 'Left Heel-Toe Ratio'] = left_ratio
    features.loc[0, 'Right Heel-Toe Ratio'] = right_ratio
    features.loc[0, 'Left - Right Single Ratio'] = single_ratio
    features.loc[0, 'Heel Ratio'] = double_heel_ratio
    features.loc[0, 'Toe Ratio'] = double_toe_ratio
    '''
    # Skewness
    features.loc[0, 'Avg Left Heel Pressure Skewness'] = left_heel_skew
    features.loc[0, 'Avg Left Toe Pressure Skewness'] = left_toe_skew
    features.loc[0, 'Left Acceleration X Skewness'] = left_accx_skew
    features.loc[0, 'Left Acceleration Y Skewness'] = left_accy_skew
    features.loc[0, 'Left Acceleration Z Skewness'] = left_accz_skew
    features.loc[0, 'Left Angular X Skewness'] = left_angx_skew
    features.loc[0, 'Left Angular Y Skewness'] = left_angy_skew
    features.loc[0, 'Left Angular Z Skewness'] = left_angz_skew
    features.loc[0, 'Left Total Force Skewness'] = left_force_skew
    features.loc[0, 'Left COP X Skewness'] = left_cntx_skew
    features.loc[0, 'Left COP Y Skewness'] = left_cnty_skew

    features.loc[0, 'Avg Right Heel Pressure Skewness'] = right_heel_skew
    features.loc[0, 'Avg Right Toe Pressure Skewness'] = right_toe_skew
    features.loc[0, 'Right Acceleration X Skewness'] = right_accx_skew
    features.loc[0, 'Right Acceleration Y Skewness'] = right_accy_skew
    features.loc[0, 'Right Acceleration Z Skewness'] = right_accz_skew
    features.loc[0, 'Right Angular X Skewness'] = right_angx_skew
    features.loc[0, 'Right Angular Y Skewness'] = right_angy_skew
    features.loc[0, 'Right Angular Z Skewness'] = right_angz_skew
    features.loc[0, 'Right Total Force Skewness'] = right_force_skew
    features.loc[0, 'Right COP X Skewness'] = right_cntx_skew
    features.loc[0, 'Right COP Y Skewness'] = right_cnty_skew

    # Kurtosis
    features.loc[0, 'Avg Left Heel Pressure Kurtosis'] = left_heel_kurtosis
    features.loc[0, 'Avg Left Toe Pressure Kurtosis'] = left_toe_kurtosis
    features.loc[0, 'Left Acceleration X Kurtosis'] = left_accx_kurtosis
    features.loc[0, 'Left Acceleration Y Kurtosis'] = left_accy_kurtosis
    features.loc[0, 'Left Acceleration Z Kurtosis'] = left_accz_kurtosis
    features.loc[0, 'Left Angular X Kurtosis'] = left_angx_kurtosis
    features.loc[0, 'Left Angular Y Kurtosis'] = left_angy_kurtosis
    features.loc[0, 'Left Angular Z Kurtosis'] = left_angz_kurtosis
    features.loc[0, 'Left Total Force Kurtosis'] = left_force_kurtosis
    features.loc[0, 'Left COP X Kurtosis'] = left_cntx_kurtosis
    features.loc[0, 'Left COP Y Kurtosis'] = left_cnty_kurtosis

    features.loc[0, 'Avg Right Heel Pressure Kurtosis'] = right_heel_kurtosis
    features.loc[0, 'Avg Right Toe Pressure Kurtosis'] = right_toe_kurtosis
    features.loc[0, 'Right Acceleration X Kurtosis'] = right_accx_kurtosis
    features.loc[0, 'Right Acceleration Y Kurtosis'] = right_accy_kurtosis
    features.loc[0, 'Right Acceleration Z Kurtosis'] = right_accz_kurtosis
    features.loc[0, 'Right Angular X Kurtosis'] = right_angx_kurtosis
    features.loc[0, 'Right Angular Y Kurtosis'] = right_angy_kurtosis
    features.loc[0, 'Right Angular Z Kurtosis'] = right_angz_kurtosis
    features.loc[0, 'Right Total Force Kurtosis'] = right_force_kurtosis
    features.loc[0, 'Right COP X Kurtosis'] = right_cntx_kurtosis
    features.loc[0, 'Right COP Y Kurtosis'] = right_cnty_kurtosis
    '''
    features.loc[0, 'Avg Left Heel Pressure Skewness'] =old_features['Avg Left Heel Pressure Skewness'].values
    features.loc[0, 'Avg Left Toe Pressure Skewness'] = old_features['Avg Left Toe Pressure Skewness'].values
    features.loc[0, 'Left Acceleration X Skewness'] = old_features['Left Acceleration X Skewness'].values
    features.loc[0, 'Left Acceleration Y Skewness'] = old_features['Left Acceleration Y Skewness'].values
    features.loc[0, 'Left Acceleration Z Skewness'] =old_features['Left Acceleration Z Skewness'].values
    features.loc[0, 'Left Angular X Skewness'] =old_features['Left Angular X Skewness'].values
    features.loc[0, 'Left Angular Y Skewness'] =old_features['Left Angular Y Skewness'].values
    features.loc[0, 'Left Angular Z Skewness'] =old_features['Left Angular Z Skewness'].values
    features.loc[0, 'Left Total Force Skewness'] =old_features['Left Total Force Skewness'].values
    features.loc[0, 'Left COP X Skewness'] =old_features['Left COP X Skewness'].values
    features.loc[0, 'Left COP Y Skewness'] =old_features['Left COP Y Skewness'].values

    features.loc[0, 'Avg Right Heel Pressure Skewness'] =old_features['Avg Right Heel Pressure Skewness'].values
    features.loc[0, 'Avg Right Toe Pressure Skewness'] =old_features['Avg Right Toe Pressure Skewness'].values
    features.loc[0, 'Right Acceleration X Skewness'] =old_features['Right Acceleration X Skewness'].values
    features.loc[0, 'Right Acceleration Y Skewness'] =old_features['Right Acceleration Y Skewness'].values
    features.loc[0, 'Right Acceleration Z Skewness'] =old_features['Right Acceleration Z Skewness'].values
    features.loc[0, 'Right Angular X Skewness'] = old_features['Right Angular X Skewness'].values
    features.loc[0, 'Right Angular Y Skewness'] =old_features['Right Angular Y Skewness'].values
    features.loc[0, 'Right Angular Z Skewness'] =old_features['Right Angular Z Skewness'].values
    features.loc[0, 'Right Total Force Skewness'] =old_features['Right Total Force Skewness'].values
    features.loc[0, 'Right COP X Skewness'] = old_features['Right COP X Skewness'].values
    features.loc[0, 'Right COP Y Skewness'] =old_features['Right COP Y Skewness'].values

    # Kurtosis
    features.loc[0, 'Avg Left Heel Pressure Kurtosis'] =old_features.loc[0, 'Avg Left Heel Pressure Kurtosis'].values
    features.loc[0, 'Avg Left Toe Pressure Kurtosis'] =old_features.loc[0, 'Avg Left Toe Pressure Kurtosis'].values
    features.loc[0, 'Left Acceleration X Kurtosis'] = old_features.loc[0, 'Left Acceleration X Kurtosis'].values
    features.loc[0, 'Left Acceleration Y Kurtosis'] =old_features.loc[0, 'Left Acceleration Y Kurtosis'].values
    features.loc[0, 'Left Acceleration Z Kurtosis'] =old_features.loc[0, 'Left Acceleration Z Kurtosis'].values
    features.loc[0, 'Left Angular X Kurtosis'] =old_features.loc[0, 'Left Angular X Kurtosis'].values
    features.loc[0, 'Left Angular Y Kurtosis'] =old_features.loc[0, 'Left Angular Y Kurtosis'].values
    features.loc[0, 'Left Angular Z Kurtosis'] =old_features.loc[0, 'Left Angular Z Kurtosis'].values
    features.loc[0, 'Left Total Force Kurtosis'] =old_features.loc[0, 'Left Total Force Kurtosis'].values
    features.loc[0, 'Left COP X Kurtosis'] =old_features.loc[0, 'Left COP X Kurtosis'].values
    features.loc[0, 'Left COP Y Kurtosis'] =old_features.loc[0, 'Left COP Y Kurtosis'].values

    features.loc[0, 'Avg Right Heel Pressure Kurtosis'] =old_features.loc[0, 'Avg Right Heel Pressure Kurtosis'].values
    features.loc[0, 'Avg Right Toe Pressure Kurtosis'] =old_features.loc[0, 'Avg Right Toe Pressure Kurtosis'].values
    features.loc[0, 'Right Acceleration X Kurtosis'] = old_features.loc[0, 'Right Acceleration X Kurtosis'].values
    features.loc[0, 'Right Acceleration Y Kurtosis'] = old_features.loc[0, 'Right Acceleration Y Kurtosis'].values
    features.loc[0, 'Right Acceleration Z Kurtosis'] =old_features.loc[0, 'Right Acceleration Z Kurtosis'].values
    features.loc[0, 'Right Angular X Kurtosis'] =old_features.loc[0, 'Right Angular X Kurtosis'].values
    features.loc[0, 'Right Angular Y Kurtosis'] =old_features.loc[0, 'Right Angular Y Kurtosis'].values
    features.loc[0, 'Right Angular Z Kurtosis'] =old_features.loc[0, 'Right Angular Z Kurtosis'].values
    features.loc[0, 'Right Total Force Kurtosis'] =old_features.loc[0, 'Right Total Force Kurtosis'].values
    features.loc[0, 'Right COP X Kurtosis'] =old_features.loc[0, 'Right COP X Kurtosis'].values
    features.loc[0, 'Right COP Y Kurtosis'] =old_features.loc[0, 'Right COP Y Kurtosis'].values

    #Approximate Entropy
    features.loc[0, 'Avg Left Heel Pressure ApEn'] = old_features['Avg Left Heel Pressure ApEn'].values
    features.loc[0, 'Avg Left Toe Pressure ApEn'] = old_features['Avg Left Toe Pressure ApEn'].values
    features.loc[0, 'Left Acceleration X ApEn'] = old_features['Left Acceleration X ApEn'].values
    features.loc[0, 'Left Acceleration Y ApEn'] = old_features['Left Acceleration Y ApEn'].values
    features.loc[0, 'Left Acceleration Z ApEn'] = old_features['Left Acceleration Z ApEn'].values
    features.loc[0, 'Left Angular X ApEn'] = old_features['Left Angular X ApEn'].values
    features.loc[0, 'Left Angular Y ApEn'] = old_features['Left Angular Y ApEn'].values
    features.loc[0, 'Left Angular Z ApEn'] = old_features['Left Angular Z ApEn'].values
    features.loc[0, 'Left Total Force ApEn'] = old_features['Left Total Force ApEn'].values
    features.loc[0, 'Left COP X ApEn'] = old_features['Left COP X ApEn'].values
    features.loc[0, 'Left COP Y ApEn'] = old_features['Left COP Y ApEn'].values

    features.loc[0, 'Avg Right Heel Pressure ApEn'] = old_features['Avg Left Heel Pressure ApEn'].values
    features.loc[0, 'Avg Right Toe Pressure ApEn'] = old_features['Avg Right Toe Pressure ApEn'].values
    features.loc[0, 'Right Acceleration X ApEn'] = old_features['Right Acceleration X ApEn'].values
    features.loc[0, 'Right Acceleration Y ApEn'] = old_features['Right Acceleration Y ApEn'].values
    features.loc[0, 'Right Acceleration Z ApEn'] = old_features['Right Acceleration Z ApEn'].values
    features.loc[0, 'Right Angular X ApEn'] = old_features['Right Angular X ApEn'].values
    features.loc[0, 'Right Angular Y ApEn'] = old_features['Right Angular Y ApEn'].values
    features.loc[0, 'Right Angular Z ApEn'] = old_features['Right Angular Z ApEn'].values
    features.loc[0, 'Right Total Force ApEn'] = old_features['Right Total Force ApEn'].values
    features.loc[0, 'Right COP X ApEn'] = old_features['Right COP X ApEn'].values
    features.loc[0, 'Right COP Y ApEn'] = old_features['Right COP Y ApEn'].values
    '''
    # Approximate Entropy
    features.loc[0, 'Avg Left Heel Pressure ApEn'] = left_heel_ae
    features.loc[0, 'Avg Left Toe Pressure ApEn'] = left_toe_ae
    features.loc[0, 'Left Acceleration X ApEn'] = left_accx_ae
    features.loc[0, 'Left Acceleration Y ApEn'] = left_accy_ae
    features.loc[0, 'Left Acceleration Z ApEn'] = left_accz_ae
    features.loc[0, 'Left Angular X ApEn'] = left_angx_ae
    features.loc[0, 'Left Angular Y ApEn'] = left_angy_ae
    features.loc[0, 'Left Angular Z ApEn'] = left_angz_ae
    features.loc[0, 'Left Total Force ApEn'] = left_force_ae
    features.loc[0, 'Left COP X ApEn'] = left_cntx_ae
    features.loc[0, 'Left COP Y ApEn'] = left_cnty_ae

    features.loc[0, 'Avg Right Heel Pressure ApEn'] = right_heel_ae
    features.loc[0, 'Avg Right Toe Pressure ApEn'] = right_toe_ae
    features.loc[0, 'Right Acceleration X ApEn'] = right_accx_ae
    features.loc[0, 'Right Acceleration Y ApEn'] = right_accy_ae
    features.loc[0, 'Right Acceleration Z ApEn'] = right_accz_ae
    features.loc[0, 'Right Angular X ApEn'] = right_angx_ae
    features.loc[0, 'Right Angular Y ApEn'] = right_angy_ae
    features.loc[0, 'Right Angular Z ApEn'] = right_angz_ae
    features.loc[0, 'Right Total Force ApEn'] = right_force_ae
    features.loc[0, 'Right COP X ApEn'] = right_cntx_ae
    features.loc[0, 'Right COP Y ApEn'] = right_cnty_ae
    '''

    feat = features.copy()
    feat.to_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/features_changed/' + name)


"""Execute"""

import os
import warnings

warnings.filterwarnings("ignore")

for root, dirs, files in os.walk('/Code/data/features_complete/'):
    if len(files) != 0:
        folder = root
        for file in files:
            patient = file.split("_")[0]
            task = file.split("_")[1].split(".")[0]
            name = patient + ' - ' + task + '.xlsx'
            if not os.path.exists(os.path.join('/Code/data/features_complete/', name)):
                print("Examining ", folder, file)
                dn_complete = pd.read_excel(folder + '/' + file)

                col = dn_complete.pop("# time")
                dn_complete.insert(0, col.name, col)
                dn_complete.drop(['Unnamed: 0', 'Event- Label level 2- Left Foot', 'Event- Label level 2- Right Foot'], axis=1, inplace=True)

                features(dn_complete, name)
            else:
                print("already done", folder, file)