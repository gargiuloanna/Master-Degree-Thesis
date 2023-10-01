from FeatureExtraction.feature_functions.utilities import get_beginning_times, get_ending_times
import numpy as np

"""
Double Support

The Double Support Time (s, %), which describes the time from the Heel Strike of the first foot until the Toe Off of the other foot.

`𝐷𝑜𝑢𝑏𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑇𝑖𝑚𝑒=(𝑇𝑜𝑒 𝑂𝑓𝑓𝑗−𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗) +(𝑇𝑜𝑒 𝑂𝑓𝑓𝑗−1−𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗−1),`

𝐷𝑜𝑢𝑏𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑃ℎ𝑎𝑠𝑒(%)=(𝐷𝑜𝑢𝑏𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑇𝑖𝑚𝑒/𝐺𝑎𝑖𝑡 𝐶𝑦𝑐𝑙𝑒)×100%.
"""


def compute_double_support_time(dn_complete, left_windows_heel, right_windows_heel, left_windows_toe, right_windows_toe):

    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    left_ending_toe = get_ending_times(left_windows_toe, dn_complete)
    right_ending_toe= get_ending_times(right_windows_toe, dn_complete)

    double_support_times = []

    stride_times = left_stride_times + right_stride_times
    stride_times.sort()
    toe_times = left_ending_toe + right_ending_toe
    toe_times.sort()

    for j in range(0, int(min(len(toe_times), len(stride_times)) / 2)):
        double_support_times.append(toe_times[j])
        double_support_times.append(stride_times[j])
    '''
    if left_windows_heel[0][0] < right_windows_heel[0][0]:
        for j in range(0, int(min(len(right_ending_toe), len(left_stride_times)) / 2)):
            double_support_times.append((left_ending_toe[j] - right_stride_times[j]))
    else:
        for j in range(0, int(min(len(left_ending_toe), len(right_stride_times)) / 2)):
            double_support_times.append(right_ending_toe[j] - left_stride_times[j])
    '''
    db_support_time = abs(np.diff(double_support_times).mean())

    return db_support_time


def compute_double_support_phase(db_support_time, gait_cycle):
    return (db_support_time / gait_cycle) * 100
