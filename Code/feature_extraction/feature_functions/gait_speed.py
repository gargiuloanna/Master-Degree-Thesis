"""
Gait Speed

The Gait Speed (m/s) is calculated by dividing the total distance covered (20 m) to the total time needed to complete the exercise.
"""

def compute_gait_speed(dn_complete, distance = 20):
    gait_speed = distance / dn_complete.loc[len(dn_complete['# time']) - 1, '# time']
    return gait_speed
