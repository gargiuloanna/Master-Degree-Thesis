from .utilities import get_beginning_times

"""Step Time, Length & Frequency

The Step Time (s), which is described as the time between two successive Heel Strikes of different foot.

𝑆𝑡𝑒𝑝 𝛵𝑖𝑚𝑒=𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗+1−𝐻𝑒𝑒𝑙 𝑆𝑡𝑟𝑖𝑘𝑒𝑗.

The Step Length (m) is calculated by dividing the total distance covered (20 m) to the total number of steps (Steps Number) which is specified as the number of Heel Strikes during gait.

`𝑆𝑡𝑒𝑝 𝐿𝑒𝑛𝑔ℎ𝑡=𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒/𝑆𝑡𝑒𝑝𝑠 𝑁𝑢𝑚𝑏𝑒𝑟`

The Step Frequency (steps/min) also called CADENCE or walking rate, describes the number of steps in the unit of time. It is given by the ratio of the steps number to the time of gait, multiplied by 60 to be expressed in minutes.

`𝑆𝑡𝑒𝑝 𝐹𝑟𝑒𝑞𝑢𝑒𝑛𝑐𝑦= (𝑆𝑡𝑒𝑝𝑠 𝑁𝑢𝑚𝑏𝑒𝑟/𝑇𝑖𝑚𝑒) × 60.`
"""
#Step modificati

def _step_time(first_heel, second_heel):
    step_times = []
    for j in range(0, min(len(first_heel), len(second_heel))):
        step_times.append(abs(second_heel[j] - first_heel[j]))
    return sum(step_times) / len(step_times)


def compute_step_time(dn_complete, left_windows_heel, right_windows_heel):

    left_stride_times = get_beginning_times(left_windows_heel, dn_complete)
    right_stride_times = get_beginning_times(right_windows_heel, dn_complete)

    step_time = _step_time(left_stride_times, right_stride_times)

    return step_time


def compute_step_length(left_windows_heel, right_windows_heel):

    steps_number = len(left_windows_heel) + len(right_windows_heel)

    left_step_length = 20/len(left_windows_heel)
    right_step_length = 20/len(right_windows_heel)
    step_length = 20 / steps_number

    return step_length, left_step_length, right_step_length

# Rivalutata solo negli intervalli di tempo in cui il paziente sta camminando, invece di valutarlo sull'intero intervallo di tempo,
# che include anche i momenti di standing.
def compute_step_frequency(dn_complete, left_windows_heel, right_windows_heel):
    left_steps_number = len(left_windows_heel)
    right_steps_number = len(right_windows_heel)

    left_step_frequency = (left_steps_number / dn_complete['# time'][left_windows_heel[-1][-1]]) * 60
    right_step_frequency = (right_steps_number / dn_complete['# time'][right_windows_heel[-1][-1]]) * 60

    step_frequency = (right_step_frequency + left_step_frequency) / 2

    return step_frequency, left_step_frequency, right_step_frequency



def step_number(left_windows_heel, right_windows_heel):

    left_steps_number = len(left_windows_heel)
    right_steps_number = len(right_windows_heel)

    step_number = int((left_steps_number + right_steps_number)/2)

    return step_number, left_steps_number, right_steps_number