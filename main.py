'''
Joel A. Jaffe 2025-03-01

This program implements the the recursive filter design from Chapter 26 of 
The Scientist and Engineer's Guide to Digital Signal Processing by Steven W. Smith, Ph.D.
https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch26.pdf

W.I.P. - This is a work in progress.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time
import textwrap

# Implements the recursive filter design algorithm from Table 26-4
class Trainer:
  def __init__(self, fft_size=1024, error_threshold = 0.01, max_poles = None, delta=0.001, mu=0.02, plot=True):
    self.fft_size = fft_size
    self.error_threshold = error_threshold
    self.num_poles = 1
    self.max_poles_flag = False if max_poles is None else True
    if max_poles is not None:
      self.max_poles = max_poles
    self.delta = delta
    self.mu = mu
    self.plot = plot
    self.target_mags = np.zeros(fft_size // 2)  # "//" ensures integer division
    self.ff_coefs = np.zeros(self.num_poles)
    self.fb_coefs = np.zeros(self.num_poles)
    self.ff_slopes = np.zeros(self.num_poles)
    self.fb_slopes = np.zeros(self.num_poles)
  
  def initCoefs(self):
    '''
    Init coefs to identity function.
    '''
    self.ff_coefs.fill(0)
    self.fb_coefs.fill(0)
    self.ff_coefs[0] = 1
    self.fb_coefs[0] = 1

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
      self.target_mags[:cutoff] = 1.1  # Passband
      self.target_mags[cutoff:] = 0.1  # Stopband

  def get_mags(self): 
    '''
    ONLY MAGS!!!
    '''
    mags = abs(scipy.signal.freqz(self.ff_coefs, self.fb_coefs, worN=self.fft_size//2, whole=False)[1])
    return mags
  
  def apply_filter(self, input_signal):
    '''
    Apply the trained filter to an input signal using scipy's lfilter
    '''
    filtered_signal = scipy.signal.lfilter(self.ff_coefs, self.fb_coefs, input_signal)
    return filtered_signal

  def calculate_error(self):
    '''
    GOSUB 3000
    '''
    mags = self.get_mags()

    # error calculation section
    error = np.sqrt(np.sum((self.target_mags[:self.fft_size // 2] - mags[:self.fft_size // 2]) ** 2) / (self.fft_size // 2))
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
      if i > 0:
        self.fb_coefs[i] -= self.mu * self.fb_slopes[i]
    
    return self.calculate_error()

  def epochs(self):
    '''
    Recursive training based on when the error stops improving.
    Trains until error doesn't improve for 3 epochs. 
    See https://www.dafx.de/paper-archive/2020/proceedings/papers/DAFx2020_paper_52.pdf
    '''
    epoch = 0
    stasis_count = 0
    curr_error = self.calculate_error()  # Init curr_error

    # main training loop
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
    
  def plot_frequency_response(self, title='Frequency Response'):
    '''
    Generates a plot to visualize trained filter vs target response
    '''
    freqs = np.linspace(0, 0.5, len(self.target_mags))  # Normalized frequency (0 to 0.5)
    trained_mags = self.get_mags()  # Get the trained magnitudes
    plt.plot(freqs, self.target_mags, 'r--', label='Target')  # Dashed line for target
    plt.plot(freqs, trained_mags, 'b-', label='Trained')  # Solid line for trained
    plt.title(f'{title} Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    legend = plt.legend()
    legend.set_title(f'num_poles: {self.num_poles}\nfft_size: {self.fft_size}')
    plt.grid(True)
    plt.show()

  def print_coefs(self):
    '''
    Print the feedforward and feedback coefficients
    '''
    print("Feedforward Coefficients (ff_coefs):")
    print(self.ff_coefs)
    print("Feedback Coefficients (fb_coefs):")
    print(self.fb_coefs)

  def generate_header(self):
    '''
    Generates a C++ header file implementing the filter as cascaded biquad sections.
    '''
    filename = 'ir-approx.h'
    guard_name = 'IR_APPROX_H'
    sos = scipy.signal.tf2sos(self.ff_coefs, self.fb_coefs)
    num_sections = sos.shape[0]
    
    content = textwrap.dedent(f'''\
    /**
    * @brief IR approximation implemented with cascaded biquad sections                          
    * @tparam T floating-point type
    */                                                    
    template <typename T>                          
    class IRApprox {{
    private:
      static constexpr unsigned N = {num_sections};
      T a[N][2] = {{{', '.join([f'{{{s[4]}, {s[5]}}}' for s in sos])}}};
      T b[N][3] = {{{', '.join([f'{{{s[0]}, {s[1]}, {s[2]}}}' for s in sos])}}};
      T w[N][2] = {{0}};            
                                      
    public:                
      IRApprox() {{
        for (unsigned i = 0; i < N; ++i) {{
          w[i][0] = 0;
          w[i][1] = 0;
        }}
      }}
      ~IRApprox() {{}}

      T processSample(const T& x0) {{
        T x = x0;
        for (unsigned i = 0; i < N; ++i) {{
          T w0 = x - a[i][0] * w[i][0] - a[i][1] * w[i][1];
          T y0 = b[i][0] * w0 + b[i][1] * w[i][0] + b[i][2] * w[i][1];
          w[i][1] = w[i][0];
          w[i][0] = w0;
          x = y0;
        }}
        return x;
      }}
    }};
    ''')
    with open(filename, "w") as f:
      f.write(f"#ifndef {guard_name}\n")
      f.write(f"#define {guard_name}\n\n")
      f.write(content + "\n")
      f.write(f"#endif // {guard_name}\n")
    print(f'Generated "{filename}"')

  def __call__(self, impulse_response=None):
    '''
    Operator to train until convergence
    '''
    # Init the model
    self.initCoefs()
    if impulse_response is not None:
      target_mags = abs(scipy.signal.freqz(impulse_response, worN=self.fft_size//2, whole=False)[1])
      self.load_target_mags(target_mags)
    else:
      self.load_target_mags() 

    # Train the model
    self.num_poles = 1
    error = 1
    start_time = time.time()  # Start timing
    if self.max_poles_flag:
      while error > self.error_threshold and self.num_poles < self.max_poles:
        self.epochs()
        error = self.calculate_error()
        self.num_poles += 1
        self.ff_coefs = np.append(self.ff_coefs, 0)
        self.fb_coefs = np.append(self.fb_coefs, 0)
        self.ff_slopes = np.append(self.ff_slopes, 0)
        self.fb_slopes = np.append(self.fb_slopes, 0)
    else:
      while error > self.error_threshold:
        self.epochs()
        error = self.calculate_error()
        self.num_poles += 1
        self.ff_coefs = np.append(self.ff_coefs, 0)
        self.fb_coefs = np.append(self.fb_coefs, 0)
        self.ff_slopes = np.append(self.ff_slopes, 0)
        self.fb_slopes = np.append(self.fb_slopes, 0)

    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate time taken
    print(f'Final error: {self.calculate_error()}')  # Print the final error
    print(f'Training time: {time_taken:.2f} seconds')  # Print the time taken
       
    if self.plot:
      self.plot_frequency_response('Target vs Trained')
      self.generate_header()

if __name__ == "__main__":
  def main():
    my_model = Trainer()
    my_model()

  main()