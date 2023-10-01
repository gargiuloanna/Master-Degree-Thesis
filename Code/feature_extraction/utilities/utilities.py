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