import numpy as np

N = 100
t = np.linspace(0,N,N)
periods = [[4,8,20],
           [3.333,13,50]]
amplitudes = [[5,3,10],
              [6,1,100]]

def makeSignal(periods,amplitudes,N):
  signal = np.zeros(N)
  for i in range(len(periods)):
    signal += np.sin(2.*np.pi/periods[i] * t) + np.cos(2.*np.pi/periods[i] * t)
  return signal

for i in range(2):
  signal = makeSignal(periods[i],amplitudes[i],N)
  with open('signal_{}.csv'.format(i),'w') as f:
    f.writelines('t,signal\n')
    for i in range(len(t)):
      f.writelines('{},{}\n'.format(t[i],signal[i]))

print signal
