import matplotlib.pyplot as plt
from Code.feature_extraction.utilities.utilities import get_signal, get_mean, get_max_min, get_last_stance, get_initial_stance, remove_stance
'''
The functions are used to plot the gait phases of each leg computed by the algorithm and compared to the labels
available in the dataset.
'''
def plots(signal, dc_signal, dn_complete, dc_complete, hes, tof, fof, her, peaks, mins, hes_dc, tof_dc, fof_dc, her_dc, mean, file, side = 'left'):
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (30,16))

  plt.suptitle("Gait Phases Visualization on the (Heel-Toe) Pressure Signal", fontweight="bold",  fontsize=16)

  ax1.set_title("Gait Phases found by the algorithm")
  ax1.plot(dn_complete['# time'],signal , label = 'Signal', color = 'r')
  ax1.plot(dn_complete['# time'][hes],signal[hes], 'o', label = 'Heel Strike', color = 'm', markersize = 4)
  ax1.plot(dn_complete['# time'][tof],signal[tof], 'o', label = 'Toe Off', color = 'g', markersize = 4)
  ax1.plot(dn_complete['# time'][fof], signal[fof], 'o', label = 'Foot Flat', color = 'y', markersize = 4)
  ax1.plot(dn_complete['# time'][her],signal[her], 'o', label = 'Heel Rise', color = 'b', markersize = 4)
  #ax1.plot(dn_complete['# time'][peaks],signal[peaks] ,"^", label = 'peaks', color = 'r')
  #ax1.plot(dn_complete['# time'][mins],signal[mins] , "v", label = 'mins', color = 'r')
  #ax1.axhline(y=0)
  #ax1.axhline(y=mean)
  ax1.set_xlabel("Time (s)")
  ax1.set_ylabel("Signal")
  ax1.legend()

  ax2.set_title("Gait Phases as found in the dataset")
  ax2.plot(dc_complete['# time'],dc_signal , label = 'Signal - Dataset', color = 'r')
  ax2.plot(dc_complete['# time'][hes_dc],dc_signal[hes_dc], 'o', label = 'Heel Strike - Dataset', color = 'm', markersize = 4)
  ax2.plot(dc_complete['# time'][tof_dc],dc_signal[tof_dc], 'o', label = 'Toe Off - Dataset', color = 'g', markersize = 4)
  ax2.plot(dc_complete['# time'][fof_dc], dc_signal[fof_dc], 'o', label = 'Foot Flat - Dataset', color = 'y', markersize = 4)
  ax2.plot(dc_complete['# time'][her_dc],dc_signal[her_dc], 'o', label = 'Heel Rise - Dataset', color = 'b', markersize = 4)
  ax2.set_xlabel("Time (s)")
  ax2.set_ylabel("Signal")
  ax2.legend()

  fig.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/gaitphasesplots/" + file.split(".")[0] +  ' - '+ side + '.png')
  plt.close()

def visualization(dn_complete, dc_complete, file):
  hes = dn_complete.loc[dn_complete['Gait Phase Left'] == 'HES'].index
  tof = dn_complete.loc[dn_complete['Gait Phase Left'] == 'TOF'].index
  her = dn_complete.loc[dn_complete['Gait Phase Left'] == 'HER'].index
  fof = dn_complete.loc[dn_complete['Gait Phase Left'] == 'FOF'].index

  hes_r = dn_complete.loc[dn_complete['Gait Phase Right'] == 'HES'].index
  tof_r = dn_complete.loc[dn_complete['Gait Phase Right'] == 'TOF'].index
  her_r = dn_complete.loc[dn_complete['Gait Phase Right'] == 'HER'].index
  fof_r = dn_complete.loc[dn_complete['Gait Phase Right'] == 'FOF'].index

  lhes_dc = dc_complete.loc[dc_complete['Event- Label level 2- Left Foot'] == 'HES'].index
  ltof_dc = dc_complete.loc[dc_complete['Event- Label level 2- Left Foot'] == 'TOF'].index
  lher_dc = dc_complete.loc[dc_complete['Event- Label level 2- Left Foot'] == 'HER'].index
  lfof_dc = dc_complete.loc[dc_complete['Event- Label level 2- Left Foot'] == 'FOF'].index

  rhes_dc = dc_complete.loc[dc_complete['Event- Label level 2- Right Foot'] == 'HES'].index
  rtof_dc = dc_complete.loc[dc_complete['Event- Label level 2- Right Foot'] == 'TOF'].index
  rher_dc = dc_complete.loc[dc_complete['Event- Label level 2- Right Foot'] == 'HER'].index
  rfof_dc = dc_complete.loc[dc_complete['Event- Label level 2- Right Foot'] == 'FOF'].index

  dc_complete['avg left heel pressure'] = dc_complete.iloc[:, 1:7].mean(axis=1)
  dc_complete['avg left toe pressure'] = dc_complete.iloc[:, 9:17].mean(axis=1)
  dc_complete['avg right heel pressure'] = dc_complete.iloc[:, 26:32].mean(axis=1)
  dc_complete['avg right toe pressure'] = dc_complete.iloc[:, 34:42].mean(axis=1)

  left_signal = get_signal(dn_complete)
  right_signal = get_signal(dn_complete, side = 'right')
  left_dc_signal = get_signal(dc_complete)
  right_dc_signal = get_signal(dc_complete, side = 'right')

  left_peaks, left_mins = get_max_min(left_signal)
  initial_stance_left, initial_stance_index_left = get_initial_stance(dn_complete, 'left')
  last_stance_left, last_stance_index_left = get_last_stance(dn_complete, 'left')
  left_peaks, left_mins = remove_stance(left_peaks, left_mins, initial_stance_index_left, last_stance_index_left)
  left_mean, signal_mean = get_mean(left_signal, initial_stance_index_left, last_stance_index_left)

  right_peaks, right_mins = get_max_min(right_signal)
  initial_stance, initial_stance_index = get_initial_stance(dn_complete, 'right')
  last_stance, last_stance_index = get_last_stance(dn_complete, 'right')
  right_peaks, right_mins = remove_stance(right_peaks, right_mins, initial_stance_index, last_stance_index)
  right_mean, signal_mean = get_mean(right_signal, initial_stance_index, last_stance_index)

  plots(left_signal, left_dc_signal, dn_complete, dc_complete, hes, tof, fof, her, left_peaks, left_mins, lhes_dc, ltof_dc, lfof_dc, lher_dc, left_mean,file, side = 'left')
  plots(right_signal, right_dc_signal, dn_complete, dc_complete, hes_r, tof_r, fof_r, her_r, right_peaks, right_mins, rhes_dc, rtof_dc, rfof_dc, rher_dc, right_mean, file, side = 'right')