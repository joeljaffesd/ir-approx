import time
from main import Trainer

import matplotlib.pyplot as plt

def benchmark_trainer():
  num_poles_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  times = []

  for num_poles in num_poles_list:
    start_time = time.time()
    
    trainer = Trainer(num_poles=num_poles, plot=False)
    trainer()
    
    end_time = time.time()
    times.append(end_time - start_time)
    print(f"num_poles: {num_poles}, time taken: {end_time - start_time} seconds")

  plt.plot(num_poles_list, times, marker='o')
  plt.xlabel('Number of Poles')
  plt.ylabel('Time Taken (seconds)')
  plt.title('Benchmark of Trainer with Increasing num_poles')
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  benchmark_trainer()