import numpy as np
from .utilities import get_beginning_times, get_ending_times

"""
Stance Time

The Stance Time (s, %), which describes the total time during a gait cycle where the foot is in contact with the ground. 
Specifically, it is described as the time where the heel of one foot, contacts the ground until the toe of the same foot 
leaves the ground.

`𝑆𝑡𝑎𝑛𝑐𝑒 𝑇𝑖𝑚𝑒=𝑇𝑜𝑒 𝑂𝑓𝑓𝑗+1−𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗,`

`𝑆𝑡𝑎𝑛𝑐𝑒 𝑃ℎ𝑎𝑠𝑒(%)=(𝑆𝑡𝑎𝑛𝑐𝑒 𝑇𝑖𝑚𝑒/𝐺𝑎𝑖𝑡 𝐶𝑦𝑐𝑙𝑒)×100%.`
"""


def _stance_time(dn_complete, toe_windows):

    toe_ending_times_stance = get_ending_times(toe_windows, dn_complete)
    toe_beginning_times_stance = get_beginning_times(toe_windows, dn_complete)

    return toe_beginning_times_stance, toe_ending_times_stance


def compute_stance_time(dn_complete, left_windows_heel, right_windows_heel, left_windows_toe, right_windows_toe):
    left_toe_beginning_times_stance, left_toe_ending_times_stance = _stance_time(dn_complete, left_windows_toe)
    right_toe_beginning_times_stance, right_toe_ending_times_stance = _stance_time(dn_complete, right_windows_toe)

    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    left_stance_times = []
    right_stance_times = []

    for idx in range(0, min(len(left_toe_ending_times_stance), len(left_stride_times))):
        left_stance_times.append(left_stride_times[idx])
        left_stance_times.append(left_toe_ending_times_stance[idx])

    for idx in range(0, min(len(right_toe_ending_times_stance), len(right_stride_times))):
        right_stance_times.append(right_stride_times[idx])
        right_stance_times.append(right_toe_ending_times_stance[idx])

    left_stance_time = np.diff(left_stance_times).mean()
    right_stance_time = np.diff(right_stance_times).mean()

    stance_time = (left_stance_time + right_stance_time) / 2

    return stance_time, left_stance_time, right_stance_time


def compute_stance_phase(stance_time, gait_cycle):
    stance_phase = (stance_time / gait_cycle) * 100
    return stance_phase