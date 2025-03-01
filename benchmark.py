import time
import matplotlib.pyplot as plt
from main import Trainer

def benchmark_trainer():
  num_poles_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120] # Hits plateau at ~100 poles
  times = []

  for num_poles in num_poles_list:
    print(f"num_poles: {num_poles}")
    start_time = time.time()
    trainer = Trainer(max_poles=num_poles, plot=False)
    trainer()
    end_time = time.time()
    times.append(end_time - start_time)

  plt.plot(num_poles_list, times, marker='o')
  plt.xlabel('Number of Poles')
  plt.ylabel('Time Taken (seconds)')
  plt.title('Benchmark of Trainer with Increasing num_poles')
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  benchmark_trainer()