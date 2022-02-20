# wakeNet: Deep Neural Networks for wind farm wake modelling and optimisation

## Abstract

<p align="justify">
Offshore wind farm modelling has been an area of rapidly increasing interest over the last two decades, with numerous analytical as well as computational-based approaches developed, in an attempt to produce designs that improve wind farm efficiency in power production. This work presents a Machine Learning (ML) framework for the rapid modelling of wind farm flow fields, using a Deep Neural Network (DNN) neural network architecture, trained here on approximate turbine wake fields, calculated on the state-of-the-art wind farm modelling software FLORIS. The constructed neural model is capable of accurately reproducing single wake deficits at hub-level for a 5MW wind turbine under yaw and a wide range of inlet hub speed and turbulence intensity conditions, at least an order of magnitude faster than the analytical wake-based solution method, yielding results with 1.5% mean absolute error. A superposition algorithm is also developed to construct flow fields over the whole wind farm domain by superimposing individual wakes. A promising advantage of the present approach is that its performance and accuracy are expected to increase even further when trained on high-fidelity CFD or real-world data through transfer learning, while its computational cost remains low.
</p>

<p align="center">
  <img src="https://github.com/soanagno/wakeNet/blob/master/dnn_fig.png">
</p>

## Publication available at:

https://iopscience.iop.org/article/10.1088/1742-6596/2151/1/012011

## FLORIS repository:

https://github.com/NREL/floris

## Requirements

* Python (3.9.7)
* FLORIS (2.4)
* Numpy (1.21.3)
* SciPy (1.7.1)
* Matplotlib (3.4.3)
* Torch (1.10.0)
* Dask (2021.10.0), optional
* Cuda (11.3), optional
