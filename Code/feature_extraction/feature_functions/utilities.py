from FeatureExtraction.utilities.utilities import get_windows

'''
This function checks that every heel strike is associated to its toe off.
It removes any other occurrences to only consider complete steps in the signal.
'''
def check_length_difference(heel, toe):
    # clean up first and last occurrences
    heel_windows = get_windows(heel)
    toe_windows = get_windows(toe)
    # ok
    if heel_windows[0][0] > toe_windows[0][0]:
        del toe_windows[0]
    # ok
    if heel[-1] > toe[-1]:
        del heel_windows[-1]

    # check throughout the signal to delete any extra occurrences
    # case: heel strike not followed by toe off
    heel_remove = []
    indx = 0
    indx_toe = 0
    while indx < min(len(toe_windows), len(heel_windows)) - 1:
        if heel_windows[indx + 1] < toe_windows[indx_toe]:
            heel_remove.append(heel_windows[indx])
            indx += 1
        else:
            indx += 1
            indx_toe += 1

    for elem in heel_remove:
        heel_windows.remove(elem)

    # case toe off not preceded by heel strike
    toe_remove = []
    indx = 0
    indx_toe = 0
    while indx_toe < min(len(toe_windows), len(heel_windows)) - 1:
        if toe_windows[indx_toe] < heel_windows[indx]:
            toe_remove.append(toe_windows[indx_toe])
            indx_toe += 1
        else:
            indx += 1
            indx_toe += 1

    for elem in toe_remove:
        toe_windows.remove(elem)

    return heel_windows, toe_windows

'''
It gets the occurrences of heel strike and toe off from the dataframe.
'''
def get_hes_toes(dn_complete, side='Left'):
    heel_strike = dn_complete.loc[dn_complete['Gait Phase ' + side] == 'HES'].index
    toe_off = dn_complete.loc[dn_complete['Gait Phase ' + side] == 'TOF'].index

    return heel_strike, toe_off

'''
This function gets the beginning instant of every window of the signal.
'''
def get_beginning_times(signal, dn_complete):
    beginning_times = []
    for window in signal:
        beginning_times.append(dn_complete['# time'][window[0]])
    return beginning_times


'''
This function gets the ending instant of every window of the signal.
'''
def get_ending_times(signal, dn_complete):
    ending_times = []
    for window in signal:
        ending_times.append(dn_complete['# time'][window[-1]])
    return ending_times