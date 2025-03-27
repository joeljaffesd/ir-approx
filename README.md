# IR-Approx: IIR Approximation of FIR Impulse Responses Using Machine Learning

This repository documents development of **IR-Approx**. 

For a more detailed introduction, see [PROPOSAL.md](./PROPOSAL.md).

<div align=center>
<img src=./media/Figure_1.png>
</div>

## Getting Started
For a basic demo, run `python main.py`, which will generate the output seen above. 

The `Trainer` class defined in `main.py` trains an IIR kernel of increasing size to approximate an impulse response (magnitudes only) until an error threshold is met. For the demo response, the algorithm takes 7.48 seconds to reach the error threshold on an M2 Max.

Run `python benchmark.py` to run the trainer with increasing `max_poles`, which demonstrates the time complexity of the training algorithm. Current results look to be O(n^2).

`notebook.ipynb` demonstrates training on a real-world impulse response (from a guitar speaker cabinet), as well as `Trainer`'s function for generating a C++ header that implements the designed filter with second-order sections.

## TODO
- Partitioning such that training on larger impulse responses won't have reduced frequency resolution 
- A time-domain training step that accounts for the delayed onset of a real-world impulse response
- Parallelization and other optimizations to reduce the time complexity of training