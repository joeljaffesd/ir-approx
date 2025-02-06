'''
Joel A. Jaffe 2025-02-03

This program implements the the recursive filter design from Chapter 26 of 
The Scientist and Engineer's Guide to Digital Signal Processing by Steven W. Smith, Ph.D.

W.I.P. - This is a work in progress.
'''

import numpy as np
import matplotlib.pyplot as plt

# Implement the recursive filter design algorithm from Table 26-4
fft_size = 256
num_poles = 8
delta = 0.00001
mu = 0.2
reals = np.zeros(fft_size)
imags = np.zeros(fft_size)
target_mags = np.zeros(fft_size // 2) # "//" ensures integer division
ff_coefs = np.zeros(num_poles)
fb_coefs = np.zeros(num_poles)
ff_slopes = np.zeros(num_poles)
fb_slopes = np.zeros(num_poles)

def initCoefs(ff_coefs, fb_coefs, filter_order):
  '''
  init coefs to identity function
  '''
  for i in range(filter_order):
    ff_coefs[i] = 0
    fb_coefs[i] = 0
  ff_coefs[0] = 1

def load_target_mags():
  '''
  GOSUB XXXX. 
  Currently draws a steep lowpass filter ala Figure 26-14a
  '''
  global target_mags
  cutoff = fft_size // 4  # Example cutoff frequency
  target_mags = np.zeros(fft_size // 2)
  target_mags[:cutoff] = 1  # Passband
  target_mags[cutoff:] = 0  # Stopband

def calculate_fft(reals, imags, fft_size):
  '''
  GOSUB 1000
  '''

  # converts data from time to frequency domain
  fft_result = np.fft.fft(reals + 1j * imags, fft_size)
  reals[:] = np.real(fft_result)[:fft_size]
  imags[:] = np.imag(fft_result)[:fft_size]

def calculate_error(ff_coefs, fb_coefs, target_mags):
  '''
  GOSUB 3000
  '''

  for i in range(fft_size):
    reals[i] = 0 # clear all bins
    imags[i] = 0  # ^
  imags[12] = 1 # ?? set imags 12 to 1 for some reason

  # looks like fast convolve for bins 12-fft_size
  for i in range(12, fft_size):
    for j in range(num_poles):
      reals[i] += (ff_coefs[j] * imags[i - j]) + (fb_coefs[j] * reals[i - j])
  imags[12] = 0 # set imags 12 to 0 for some reason

  calculate_fft(reals, imags, fft_size) # GOSUB 1000

  # error calculation section
  error = 0
  for i in range(fft_size // 2):  # Use integer division
    mag = np.sqrt(reals[i] ** 2 + imags[i] ** 2)
    error += (mag - target_mags[i]) ** 2
  error = np.sqrt(error / ((fft_size // 2) + 1))

  return error

def train(ff_coefs, fb_coefs, delta, mu, prev_error):
  '''
  GOSUB 2000.
  Implements the training algorithm from Table 26-5.
  Returns error after forward pass.
  '''
  for i in range(num_poles):
    ff_coefs[i] += delta
    error = calculate_error(ff_coefs, fb_coefs, target_mags)
    ff_slopes[i] = (error - prev_error) / delta
    ff_coefs[i] -= delta
    if i > 0:
      fb_coefs[i] += delta
      error = calculate_error(ff_coefs, fb_coefs, target_mags)
      fb_slopes[i] = (error - prev_error) / delta
      fb_coefs[i] -= delta

  for i in range(num_poles):
    ff_coefs[i] -= mu * ff_slopes[i]
    fb_coefs[i] -= mu * fb_slopes[i]
  
  return calculate_error(ff_coefs, fb_coefs, target_mags)

def epochs(mu):
  '''
  Recursive training based on when the error stops improving.
  Trains until error doesn't improve for 100 epochs. 
  See https://www.dafx.de/paper-archive/2020/proceedings/papers/DAFx2020_paper_52.pdf
  '''
  epoch = 0
  stasis_count = 0
  curr_error = calculate_error(ff_coefs, fb_coefs, target_mags)  # Init curr_error

  while(stasis_count < 100):  # While the new error is less than the current error
    epoch += 1  # Increment epoch
    new_error = train(ff_coefs, fb_coefs, delta, mu, curr_error)  # Call train and get the new error
    if new_error > curr_error: # If the error increased, reduce the step size
      mu /= 2
    if abs(new_error - curr_error) < 1e-6:  # If the improvement is less than a small threshold
      stasis_count += 1  # Increment the stasis count
    else:
      stasis_count = 0  # Otherwise, reset the stasis count
    curr_error = new_error  # Set the current error to the new error

  print(f'Final error: {curr_error}')  # Print the final error
  print(f'Epochs: {epoch}')  # Print the number of epochs

def plot_frequency_response(target_mags, trained_mags, title='Frequency Response'):
  '''
  Generates a plot to visualize trained filter vs target response
  '''
  freqs = np.linspace(0, 0.5, len(target_mags))  # Normalized frequency (0 to 0.5)
  plt.plot(freqs, target_mags, 'r--', label='Target')  # Dashed line for target
  plt.plot(freqs, trained_mags, 'b-', label='Trained')  # Solid line for trained
  plt.title(f'{title} Frequency Response')
  plt.xlabel('Normalized Frequency')
  plt.ylabel('Magnitude')
  plt.legend()
  plt.grid(True)
  plt.show()

def main():
  initCoefs(ff_coefs, fb_coefs, num_poles)
  load_target_mags() # GOSUB XXXX
  epochs(mu) # Train the model

  # Calculate the frequency response of the trained model
  calculate_fft(reals, imags, fft_size)
  trained_mags = np.sqrt(reals[:fft_size // 2] ** 2 + imags[:fft_size // 2] ** 2)

  # Plot Results
  plot_frequency_response(target_mags, trained_mags, 'Target vs Trained')

main()