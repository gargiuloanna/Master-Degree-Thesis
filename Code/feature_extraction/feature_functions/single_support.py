from .utilities import get_beginning_times, get_ending_times
import numpy as np

"""
Single Support

The Single Support Time (s, %), which describes the time from the Toe Off of the one foot until the Heel Strike of the other foot.

`𝑆𝑖𝑛𝑔𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑇𝑖𝑚𝑒=(𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗−𝑇𝑜𝑒 𝑂𝑓𝑓𝑗−1 +(𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗+1−𝑇𝑜𝑒 𝑂𝑓𝑓𝑗),`

`𝑆𝑖𝑛𝑔𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑃ℎ𝑎𝑠𝑒(%)=(𝑆𝑖𝑛𝑔𝑙𝑒 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 𝑇𝑖𝑚𝑒/𝐺𝑎𝑖𝑡 𝐶𝑦𝑐𝑙𝑒)×100%.`
"""


def _support_time(dn_complete, windows_toe):
    toe_ending_times = get_ending_times(windows_toe, dn_complete)
    toe_beginning_times = get_beginning_times(windows_toe, dn_complete)

    return toe_ending_times, toe_beginning_times


def compute_single_support_time(dn_complete, left_windows_toe, right_windows_toe, left_windows_heel, right_windows_heel):

    left_toe_ending_times, left_toe_beginning_times = _support_time(dn_complete, left_windows_toe)
    right_toe_ending_times, right_toe_beginning_times = _support_time(dn_complete, right_windows_toe)


    left_stride_times = get_ending_times(left_windows_heel, dn_complete)
    right_stride_times = get_ending_times(right_windows_heel, dn_complete)


    stride_times = left_stride_times + right_stride_times
    stride_times.sort()
    toe_times = left_toe_beginning_times + right_toe_beginning_times
    toe_times.sort()

    single_support_times = []

    for j in range(1, int((min(len(stride_times), len(toe_times)) - 1) / 2)):
        single_support_times.append(abs(stride_times[j+1] - toe_times[j]) + abs(stride_times[j] - toe_times[j-1]))
    '''
    single_support_times = []
    if left_windows_toe[0][0] < right_windows_toe[0][0]:
        for j in range(0, int( (min(len(left_toe_beginning_times), len(right_stride_times)) - 1) /2)):
            single_support_times.append((right_stride_times[j] - left_toe_beginning_times[j]))
    else:
        for j in range(0, int( (min(len(right_toe_beginning_times), len(left_stride_times)) - 1) /2)):
            single_support_times.append((left_stride_times[j] - right_toe_beginning_times[j]))
    '''
    single_support_time = sum(single_support_times) / len(single_support_times)

    return single_support_time


def compute_single_support_phase(single_support_time, gait_cycle):
    return (single_support_time / gait_cycle) * 100