'''
Joel A. Jaffe 2025-02-03

This program implements the the recursive filter design from Chapter 26 of 
The Scientist and Engineer's Guide to Digital Signal Processing by Steven W. Smith, Ph.D.

W.I.P. - This is a work in progress.
'''

import numpy as np
import matplotlib.pyplot as plt
import time

# Implements the recursive filter design algorithm from Table 26-4
class Trainer:
  def __init__(self, fft_size=256, num_poles=8, delta=0.0001, mu=0.2, plot=True):
    self.fft_size = fft_size
    self.num_poles = num_poles
    self.delta = delta
    self.mu = mu
    self.plot = plot
    self.reals = np.zeros(fft_size)
    self.imags = np.zeros(fft_size)
    self.target_mags = np.zeros(fft_size // 2)  # "//" ensures integer division
    self.ff_coefs = np.zeros(num_poles)
    self.fb_coefs = np.zeros(num_poles)
    self.ff_slopes = np.zeros(num_poles)
    self.fb_slopes = np.zeros(num_poles)
  
  def initCoefs(self):
    '''
    Init coefs to identity function.
    '''
    for i in range(self.num_poles):
      self.ff_coefs[i] = 0
      self.fb_coefs[i] = 0
    self.ff_coefs[0] = 1

  def load_target_mags(self, target_mags=None):
    '''
    `GOSUB XXXX`. 
    Loads target magnitudes. If no target magnitudes are provided, 
    it draws a steep lowpass filter ala Figure 26-14a.
    '''
    if target_mags is not None:
      if len(target_mags) != self.fft_size // 2:
        raise ValueError(f"target_mags must be of length {self.fft_size // 2}")
      self.target_mags = target_mags
    else:
      cutoff = self.fft_size // 4  # Example cutoff frequency
      self.target_mags = np.zeros(self.fft_size // 2)
      self.target_mags[:cutoff] = 1  # Passband
      self.target_mags[cutoff:] = 0  # Stopband

  def calculate_fft(self, reals, imags, fft_size):
    '''
    `GOSUB 1000`.
    Converts data from time to frequency domain.
    '''
    fft_result = np.fft.fft(reals + 1j * imags, fft_size)
    reals[:] = np.real(fft_result)[:fft_size]
    imags[:] = np.imag(fft_result)[:fft_size]

  def calculate_error(self):
    '''
    GOSUB 3000
    '''

    # clear all bins
    for i in range(self.fft_size):
      self.reals[i] = 0 
      self.imags[i] = 0
    self.imags[12] = 1 # ?? set imags[12] to 1 for some reason

    # looks like fast convolve for bins 12-fft_size
    for i in range(12, self.fft_size):
      for j in range(self.num_poles):
        self.reals[i] += (self.ff_coefs[j] * self.imags[i - j]) + (self.fb_coefs[j] * self.reals[i - j])
    self.imags[12] = 0 # set imags[12] to 0 for some reason

    self.calculate_fft(self.reals, self.imags, self.fft_size) # GOSUB 1000

    # error calculation section
    error = 0
    for i in range(self.fft_size // 2):  # Use integer division
      mag = np.sqrt(self.reals[i] ** 2 + self.imags[i] ** 2)
      error += (mag - self.target_mags[i]) ** 2
    error = np.sqrt(error / ((self.fft_size // 2) + 1))

    return error

  def forward_pass(self, prev_error):
    '''
    GOSUB 2000.
    Implements the training algorithm from Table 26-5.
    Returns error after forward pass.
    '''
    for i in range(self.num_poles):
      self.ff_coefs[i] += self.delta
      error = self.calculate_error()
      self.ff_slopes[i] = (error - prev_error) / self.delta
      self.ff_coefs[i] -= self.delta
      if i > 0:
        self.fb_coefs[i] += self.delta
        error = self.calculate_error()
        self.fb_slopes[i] = (error - prev_error) / self.delta
        self.fb_coefs[i] -= self.delta

    for i in range(self.num_poles):
      self.ff_coefs[i] -= self.mu * self.ff_slopes[i]
      self.fb_coefs[i] -= self.mu * self.fb_slopes[i]
    
    return self.calculate_error()

  def epochs(self):
    '''
    Recursive training based on when the error stops improving.
    Trains until error doesn't improve for 100 epochs. 
    See https://www.dafx.de/paper-archive/2020/proceedings/papers/DAFx2020_paper_52.pdf
    '''
    epoch = 0
    stasis_count = 0
    curr_error = self.calculate_error()  # Init curr_error

    start_time = time.time()  # Start timing

    while(stasis_count < 3):  # While the new error is less than the current error
      epoch += 1  # Increment epoch
      new_error = self.forward_pass(curr_error)  # Call train and get the new error
      if new_error > curr_error: # If the error increased, reduce the step size
        self.mu /= 2
      if abs(new_error - curr_error) < self.delta:  # If the improvement is less than a small threshold
        stasis_count += 1  # Increment the stasis count
      else:
        stasis_count = 0  # Otherwise, reset the stasis count
      curr_error = new_error  # Update curr_error

    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate time taken

    if self.plot:
      print(f'Final error: {curr_error}')  # Print the final error
      print(f'Epochs: {epoch}')  # Print the number of epochs
      print(f'Time taken: {time_taken:.2f} seconds')  # Print the time taken
    
  def plot_frequency_response(self, title='Frequency Response'):
    '''
    Generates a plot to visualize trained filter vs target response
    '''
    freqs = np.linspace(0, 0.5, len(self.target_mags))  # Normalized frequency (0 to 0.5)
    trained_mags = np.sqrt(self.reals[:self.fft_size // 2] ** 2 + self.imags[:self.fft_size // 2] ** 2)
    plt.plot(freqs, self.target_mags, 'r--', label='Target')  # Dashed line for target
    plt.plot(freqs, trained_mags, 'b-', label='Trained')  # Solid line for trained
    plt.title(f'{title} Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

  def __call__(self, target_mags=None):
    '''
    Operator to train until convergence
    '''
    # Train the model
    self.initCoefs()
    if target_mags is not None:
      self.load_target_mags(target_mags)
    else:
      self.load_target_mags() 
    self.epochs() 

    # Create an impulse
    impulse = np.zeros(self.fft_size)
    impulse[0] = 1  

    # Filter the impulse using the trained coefficients (feedforward first via convolution)
    filtered_signal = np.convolve(impulse, self.ff_coefs, mode='full')[:self.fft_size]

    # Apply feedback coefficients (directly in a loop)
    for i in range(1, self.fft_size):
        for j in range(1, min(self.num_poles, i + 1)):
            filtered_signal[i] += self.fb_coefs[j] * filtered_signal[i - j]

    # Compute FFT of the filtered impulse response
    reals = np.zeros(self.fft_size)
    imags = np.zeros(self.fft_size)
    self.calculate_fft(filtered_signal, imags, self.fft_size)  # Apply FFT

    trained_mags = np.sqrt(self.reals[:self.fft_size // 2] ** 2 + self.imags[:self.fft_size // 2] ** 2)
    if self.plot:
      self.plot_frequency_response('Target vs Trained')

if __name__ == "__main__":
  def main():
    my_model = Trainer()
    my_model()

  main()