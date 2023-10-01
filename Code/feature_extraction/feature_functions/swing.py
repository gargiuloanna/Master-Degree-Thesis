from FeatureExtraction.feature_functions.utilities import get_beginning_times

"""
Swing Time & Phase

The Swing Time (s, %), which describes the time from the Toe Off of the one foot until the Heel Strike of the same foot.

𝑆𝑤𝑖𝑛𝑔 𝑇𝑖𝑚𝑒=𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗+1−𝑇𝑜𝑒 𝑂𝑓𝑓𝑗,

𝑆𝑤𝑖𝑛𝑔 𝑃ℎ𝑎𝑠𝑒(%)=(𝑆𝑤𝑖𝑛𝑔 𝑇𝑖𝑚𝑒/𝐺𝑎𝑖𝑡 𝐶𝑦𝑐𝑙𝑒)×100%.
"""


def _swing_time(toe, heel):
    swing_times = []
    for j in range(0, min(len(toe), len(heel)) - 1):
        swing_times.append(heel[j + 1] - toe[j])
    return sum(swing_times) / len(swing_times)


def compute_swing_time(dn_complete, left_windows_toe, right_windows_toe, left_windows_heel, right_windows_heel):

    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    left_toe_beginning_times =get_beginning_times(left_windows_toe, dn_complete)
    right_toe_beginning_times = get_beginning_times(right_windows_toe, dn_complete)

    swing_time_left = _swing_time(left_toe_beginning_times, left_stride_times)
    swing_time_right = _swing_time(right_toe_beginning_times, right_stride_times)

    swing_time = (swing_time_right + swing_time_left) / 2

    return swing_time, swing_time_left, swing_time_right


def compute_swing_phase(swing_time, gait_cycle):
    return (swing_time / gait_cycle) * 100