# IR-Approx: IIR Approximation of FIR Impulse Responses Using Neural Networks

This repository documents development of **IR-Approx**. For our proposal, see [PROPOSAL.md](./PROPOSAL.md). For the initial prototype, see our [Colab Notebook](https://colab.research.google.com/drive/1jvvUkCaEiVgBp3HYy_lhlEEdbZr5T1Lq?usp=sharing).

### Block Diagram
```mermaid
flowchart TD

step1["`IR of size $$N_{IR}$$`"]
step1 --> step2["`IIR Kernel of size $$\frac{N_{IR}}{? \gt 1}$$`"]
step2 --> step3["NN Optimization"]
step3 --> step4["Good 'nuff?"]
step4 --noo--> no["$$?$$--"] --> step2
step4 --yes--> step5["Done!"]
```