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
reals = np.zeros(fft_size - 1)
imags = np.zeros(fft_size - 1)
target_mags = np.zeros(fft_size // 2) # "//" ensures integer division
ff_coefs = np.zeros(num_poles)
fb_coefs = np.zeros(num_poles)
ff_slopes = np.zeros(num_poles)
fb_slopes = np.zeros(num_poles)

def load_target_mags():
    global target_mags
    # Create a simple lowpass filter target magnitude response
    cutoff = fft_size // 8  # Example cutoff frequency
    target_mags = np.zeros(fft_size // 2)
    target_mags[:cutoff] = 1  # Passband
    target_mags[cutoff:] = 0  # Stopband

load_target_mags() # GOSUB XXXX

def calculate_fft(reals, imags, fft_size):
    # Implement or import the FFT calculation
    # Example using numpy's FFT
    fft_result = np.fft.fft(reals + 1j * imags, fft_size)
    reals[:] = np.real(fft_result)[:fft_size - 1]
    imags[:] = np.imag(fft_result)[:fft_size - 1]

def calculate_error(ff_coefs, fb_coefs, target_mags):
  
  for i in range(fft_size- 1):
    reals[i] = 0
    imags[i] = 0
  imags[12] = 1 # ??

  for i in range(12, fft_size - 1):
    for j in range(num_poles):
      reals[i] += (ff_coefs[j] * imags[i - j]) + (fb_coefs[j] * reals[i - j])
  imags[12] = 0 # ??

  calculate_fft(reals, imags, fft_size) # GOSUB 1000

  error = 0
  for i in range(fft_size // 2):  # Use integer division
    mag = np.sqrt(reals[i] ** 2 + imags[i] ** 2)
    error += (mag - target_mags[i]) ** 2
  error = np.sqrt(error / ((fft_size // 2) + 1))
  return error

# init to identity function
for i in range(num_poles):
    ff_coefs[i] = 0
    fb_coefs[i] = 0
ff_coefs[0] = 1

# Implement the training algorithms from Table 26-5
def train(ff_coefs, fb_coefs, delta, mu):

  error = calculate_error(ff_coefs, fb_coefs, target_mags)
  prev_error = error 

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

num_iters = 100 # currently a bottleneck. Need to get GPU acceleration working
for i in range(num_iters):
  curr_error = calculate_error(ff_coefs, fb_coefs, target_mags)  # Store the current error
  new_error = train(ff_coefs, fb_coefs, delta, mu)  # Call train and get the new error
  if new_error > curr_error: # If the error increased, reduce the step size
    mu /= 2

def plot_frequency_response(target_mags, trained_mags, title='Frequency Response'):
  freqs = np.linspace(0, 0.5, len(target_mags))  # Normalized frequency (0 to 0.5)
  plt.plot(freqs, target_mags, 'r--', label='Target')  # Dashed line for target
  plt.plot(freqs, trained_mags, 'b-', label='Trained')  # Solid line for trained
  plt.title(f'{title} Frequency Response')
  plt.xlabel('Normalized Frequency')
  plt.ylabel('Magnitude')
  plt.legend()
  plt.grid(True)
  plt.show()

# Calculate the frequency response of the trained model
calculate_fft(reals, imags, fft_size)
trained_mags = np.sqrt(reals[:fft_size // 2] ** 2 + imags[:fft_size // 2] ** 2)

# Example usage
plot_frequency_response(target_mags, trained_mags, 'Target vs Trained')