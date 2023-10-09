import math
from Code.const import g


class DimensionlessEquations:

    # stride length, step length or step width
    def length(self, length_raw, height):
        length_feature = length_raw / height
        return length_feature

    def gait_cadence(self, gait_cadence_raw,height):
        gait_cadence = gait_cadence_raw / (60 * math.sqrt(g/height))
        return gait_cadence

    #double support time, stance time, swing time, step time, or stride time
    def gait_time(self,time_raw, height):
        gait = time_raw / math.sqrt(height/g)
        return gait




