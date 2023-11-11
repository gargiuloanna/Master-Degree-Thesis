# -*- coding: utf-8 -*-
import os
import pandas as pd
from Code.feature_extraction.utilities.preprocessing import clean_data, impute, compute_average, scale
from Code.feature_extraction.utilities.utilities import get_signal, get_windows, get_mean, get_max_min, get_last_stance, get_initial_stance, remove_stance
from Code.feature_extraction.utilities.plotting import visualization

'''
Computes the gait phases associated to the signal.
'''
def gait_phases(signal, dn_complete, side = "Left"):
  #get initial information
  peaks, mins = get_max_min(signal)
  initial_stance, initial_stance_index = get_initial_stance(dn_complete, side.lower())
  last_stance, last_stance_index = get_last_stance(dn_complete, side.lower())
  peaks, mins = remove_stance(peaks, mins, initial_stance_index, last_stance_index)
  #the mean of the signal as the patient is walking is used to determine the beginning of the heel strike phase, as the first point above the mean
  mean, signal_mean = get_mean(signal, initial_stance_index, last_stance_index)
  above_mean_index = get_windows(signal_mean[signal_mean>mean].index)

  mins_index = 0
  peaks_index = 0
  mean__window_index = 0
  signal_index = 0
  update = True
  while signal_index < len(dn_complete):
    #1 --> fill in the values in the stance phase
    if signal_index < initial_stance_index:
      #print("Putting label FOF in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
      dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
      signal_index +=1

    if signal_index >= initial_stance_index and signal_index <= last_stance_index and signal_index < len(dn_complete):

      if signal[signal_index] < mean:
        #print("Signal below mean")
        #print("peaks, mins", peaks[peaks_index], mins[mins_index])
        if mins[mins_index]< peaks[peaks_index]: #heel rise
          while signal_index <= mins[mins_index] and signal_index < len(dn_complete):
            #print("HER in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
            dn_complete.loc[signal_index, "Gait Phase " + side] = 'HER'
            signal_index +=1
          if mins_index + 1 < len(mins):
              mins_index +=1
          else:
            while signal_index < len(dn_complete):
              dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
              #print("Putting label FOF in position ", signal_index, "END OF SIGNAL, after her")
              signal_index +=1
            return 0

        if peaks[peaks_index] < mins[mins_index]:
          #print("TOF in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
          #print("above mean index ", above_mean_index[mean__window_index][0])
          while signal_index < len(dn_complete) and (signal_index <= above_mean_index[mean__window_index][0] or signal[signal_index] <0):
            if signal_index < peaks[peaks_index]:
              dn_complete.loc[signal_index, "Gait Phase " + side] = 'TOF'
              signal_index +=1
              update = True
            else:
              update = False
              signal_index +=1
              if peaks_index + 1 < len(peaks):
                peaks_index +=1
              else:
                while signal_index < len(dn_complete):
                  dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
                  #print("Putting label FOF in position ", signal_index, "END OF SIGNAL, after tof")
                  signal_index +=1
                return 0
          #print("END TOF in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
          if mean__window_index + 1 < len(above_mean_index) and update:
            mean__window_index +=1

      if signal[signal_index] >= mean or signal[signal_index] >= 0:
        #print("Signal above mean")
        #print("peaks, mins", peaks[peaks_index], mins[mins_index])

        if mins[mins_index]<peaks[peaks_index]: #fof
          #print("signal index, signal, mean ", signal_index, signal[signal_index], mean)
          while signal_index < len(dn_complete) and (signal[signal_index] >= mean or signal[signal_index] >= 0):
            #print("FOF in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
            dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
            if signal_index == peaks[peaks_index]:
              if peaks_index + 1 < len(peaks):
                peaks_index +=1
              else:
                while signal_index < len(dn_complete):
                  dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
                  #print("Putting label FOF in position ", signal_index, "END OF SIGNAL")
                  signal_index +=1
                return 0
            if signal_index == mins[mins_index]:
              if mins_index + 1 < len(mins):
                mins_index +=1
              else:
                  while signal_index < len(dn_complete):
                    dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
                    #print("Putting label FOF in position ", signal_index, "END OF SIGNAL, after fof")
                    signal_index +=1
                  return 0
            if signal_index +1 < len(dn_complete):
              signal_index +=1


        if peaks[peaks_index] < mins[mins_index]: #hes

          while signal_index <= peaks[peaks_index]:
            #print("HES in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
            dn_complete.loc[signal_index, "Gait Phase " + side] = 'HES'
            signal_index +=1
          if peaks_index + 1 < len(peaks):
            peaks_index +=1
          else:
            while signal_index < len(dn_complete):
              dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
              #print("Putting label FOF in position ", signal_index, "END OF SIGNAL, after HES")
              signal_index += 1
            return 0

    #4 --> fill in the values in stance phase
    if signal_index >= last_stance_index:
      #print("Putting label FOF in position ", signal_index, "with value ", signal[signal_index].round(2), " with time ", dn_complete.loc[signal_index, '# time'])
      dn_complete.loc[signal_index, "Gait Phase " + side] = 'FOF'
      if signal_index +1 < len(dn_complete):
        signal_index +=1

  return 0



if __name__ == '__main__':

  for root, dirs, files in os.walk('/Code/data/SmartInsole/'):
    for file in files:
      print("Preprocessing file ", file)
      if not os.path.exists('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/data/gaitphasesplots/' + file.split(".")[0] + " - left.png") and not file== "s006_1slow1.xlsx" and not file == "s013_2tug1.xlsx" and not file == "s013_2tug2.xlsx":
        dc_complete = pd.read_excel(root + '/' + file)
        dn_complete = pd.read_excel(root + '/' + file)

        dn_complete = clean_data(dn_complete)
        dn_complete = impute(dn_complete)
        dn_complete = compute_average(dn_complete)
        dn_complete = scale(dn_complete)

        print("Left")
        left_signal = get_signal(dn_complete, side = 'left')
        gait_phases(left_signal, dn_complete, side="Left")

        print("Right")
        right_signal = get_signal(dn_complete, side = 'right')
        gait_phases(right_signal, dn_complete, side="Right")

        print("Visualization")
        visualization(dn_complete, dc_complete, file)

        print("Saving")
        #dn_complete.to_excel('C:/Users/annin/PycharmProjects/Tesi/data/gaitphases/' +  file)
      else:
        print("Already done ", file)

