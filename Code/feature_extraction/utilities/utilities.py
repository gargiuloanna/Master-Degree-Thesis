from pyampd.ampd import find_peaks

'''
This function splits the indices of the signal based on their difference:
consecutive indices are associated with a window, whereas when they're not consecutive the window ends.
'''
def get_windows(signal):
  windows = []
  window = []
  window.append(signal[0])
  for indx in range(1, len(signal)):
    if signal[indx] -  signal[indx-1] == 1:
      window.append(signal[indx])
    else:
      windows.append(window)
      window = []
      window.append(signal[indx])
  windows.append(window)

  return windows

'''
It returns the peaks and minimums of the signal, as indices.
'''
def get_max_min(signal):
  peaks = find_peaks(signal, scale=100)
  mins = find_peaks(-signal, scale=100)

  return peaks, mins

'''
It returns the mean of the signal registered DURING WALKING
'''
def get_mean(signal, initial_stance_index, last_stance_index):
  signal_mean = signal.loc[initial_stance_index:last_stance_index]
  mean = signal_mean.mean()
  return mean, signal_mean

'''
It determines the value of the signal corresponding to the difference between the heel pressure and toe pressure.
The output is a signal whose maximums are the peaks of the heel signal, and the minimums are the peak of the toe signal.
'''
def get_signal(dn_complete, side = 'left'):
  return dn_complete['avg ' + side + ' heel pressure'] - dn_complete['avg ' + side + ' toe pressure']


'''
Computes the initial stance time of the patient looking at the angular acceleration along the X axis.
It calculates the mean and standard deviation of the signal to determine a threshold to use to define the end
of the stance phase.
'''
def get_initial_stance(dn_complete, side = 'left'):
  plus_stance = dn_complete[side + ' angular X[dps]'].mean() + dn_complete[side + ' angular X[dps]'].std()
  minus_stance = dn_complete[side + ' angular X[dps]'].mean() - dn_complete[side + ' angular X[dps]'].std()
  previous = True
  stance_time = 0.00
  stance_index = 0
  for elem in range(len(dn_complete)):
    if (dn_complete.iloc[elem][side + ' angular X[dps]'] < plus_stance and dn_complete.iloc[elem][side + ' angular X[dps]'] > minus_stance) and previous == True:
      stance_time = dn_complete.iloc[elem]['# time']
      stance_index = elem
    else:
      previous = False
  return stance_time, stance_index

'''
Computes the final stance time of the patient looking at the angular acceleration along the X axis.
It calculates the mean and standard deviation of the signal to determine a threshold to use to define the end
of the stance phase.
'''
def get_last_stance(dn_complete, side = 'left'):
  plus_stance = dn_complete[side + ' angular X[dps]'].mean() + dn_complete[side + ' angular X[dps]'].std()
  minus_stance = dn_complete[side + ' angular X[dps]'].mean() - dn_complete[side + ' angular X[dps]'].std()
  stance_time = dn_complete.loc[len(dn_complete)- 1, '# time']
  stance_index = len(dn_complete)- 1
  last = True
  for elem in range(len(dn_complete)-1, 0, -1):
    if (dn_complete.iloc[elem][side + ' angular X[dps]'] < plus_stance and dn_complete.iloc[elem][side + ' angular X[dps]'] > minus_stance) and last == True:
      stance_time = dn_complete.iloc[elem]['# time']
      stance_index = elem
    else:
      last = False
  return stance_time, stance_index

'''
It removes from the signal the values that would be computed to determine the gait phases during stance time,
since the insoles still register a pressure as the patient is standing, but this pressure shall not be used
to determine walking gait phases.
'''
def remove_stance(peaks, mins, initial_stance_index, last_stance_index):

  #remove elements in the stance phase
  mask = peaks >= initial_stance_index
  peaks = peaks[mask]

  mask = mins >= initial_stance_index
  mins = mins[mask]


  mask = peaks <= last_stance_index
  peaks = peaks[mask]

  mask = mins <= last_stance_index
  mins = mins[mask]

  return peaks, mins
