import numpy as np
from .utilities import get_beginning_times

"""
Stride Time, Length, Velocity

The Stride Time (s), which is equal to the time between two successive Heel Strikes of the same foot.

`𝑆𝑡𝑟𝑖𝑑𝑒 𝛵𝑖𝑚𝑒=𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗+2−𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗`


Gait Variability refers to the difference between the duration of the strides.

The Stride Length (m) is calculated by dividing the total distance covered (20 m) to the total number of strides (Strides Number).

𝑆𝑡𝑟𝑖𝑑𝑒 𝐿𝑒𝑛𝑔ℎ𝑡=𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒/𝑆𝑡𝑟𝑖𝑑𝑒𝑠 𝑁𝑢𝑚𝑏𝑒𝑟.

The Gait Velocity (m/s), which describes the displacement in the unit of time, is given by the ratio of the total distance to the total time, or by the ratio of the mean values of stride length to stride time.

`𝐺𝑎𝑖𝑡 𝑉𝑒𝑙𝑜𝑐𝑖𝑡𝑦=𝑆𝑡𝑟𝑖𝑑𝑒 𝐿𝑒𝑛𝑔𝑡ℎ/𝑆𝑡𝑟𝑖𝑑𝑒 𝑇𝑖𝑚𝑒.`
"""


def compute_stride_time(dn_complete, left_windows_heel, right_windows_heel):
    # get the beginning of the stride occurrences
    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    # The stride time is equal to the time between two consecutive stride times.
    # To obtain a single value, the differences are averaged.
    left_stride_time = np.diff(left_stride_times).mean()
    right_stride_time = np.diff(right_stride_times).mean()

    # Gait Cycle represents the time needed by the patient to complete a full gait cycle
    gait_cycle = (left_stride_time + right_stride_time) / 2

    return gait_cycle, left_stride_time, right_stride_time


def compute_stride_length(left_windows_heel, right_windows_heel):
    left_strides_number = len(np.diff(np.arange(len(left_windows_heel))))
    left_stride_length = 20 / left_strides_number

    right_strides_number = len(np.diff(np.arange(len(right_windows_heel))))
    right_stride_length = 20 / right_strides_number

    stride_length = (right_stride_length + left_stride_length) / 2

    return stride_length, left_stride_length, right_stride_length


def compute_stride_velocity(left_stride_length, right_stride_length, left_stride_time, right_stride_time):
    left_gait_velocity = left_stride_length / left_stride_time
    right_gait_velocity = right_stride_length / right_stride_time

    gait_velocity = (left_gait_velocity + right_gait_velocity) / 2

    return gait_velocity, left_gait_velocity, right_gait_velocity

def compute_gait_variability(dn_complete, left_windows_heel, right_windows_heel):
    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    left_gait_variability = np.diff(left_stride_times).std()
    right_gait_variability = np.diff(right_stride_times).std()

    gait_variability = (left_gait_variability + right_gait_variability) / 2

    return gait_variability, left_gait_variability, right_gait_variability
