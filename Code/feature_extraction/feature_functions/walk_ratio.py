"""
Walk Ratio

The Walk Ratio (mm/step/min) represents the relationship between the width (base of gait) and the frequency of steps and is given by the ratio of step length to Step Frequency.

`𝑊𝑎𝑙𝑘 𝑅𝑎𝑡𝑖𝑜=𝑆𝑡𝑒𝑝 𝐿𝑒𝑛𝑔𝑡ℎ/𝑆𝑡𝑒𝑝 𝐹𝑟𝑒𝑞𝑢𝑒𝑛𝑐𝑦`.
"""

def compute_walk_ratio(left_step_length, left_step_frequency, right_step_length, right_step_frequency):
    left_walk_ratio = left_step_length / left_step_frequency
    right_walk_ratio = right_step_length / right_step_frequency

    walk_ratio = (right_walk_ratio + left_walk_ratio) / 2

    return walk_ratio, left_walk_ratio, right_walk_ratio