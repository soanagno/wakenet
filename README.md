# wakeNet: Deep Neural Networks for wind farm wake modelling and optimisation

## Abstract

<p align="justify">
Wind farm modelling is an area of rapidly increasing interest with numerous analytical and computational-based approaches developed to extend the margins of wind farm efficiency and maximise power production. In this work, we present the novel ML framework WakeNet, which reproduces generalised 2D turbine wake velocity fields at hub-height, with a mean accuracy of 99.8% compared to the solution calculated by the state-of-the-art wind farm modelling software FLORIS. As the generation of sufficient high-fidelity data for network training purposes can be cost-prohibitive, the utility of multi-fidelity transfer learning has also been investigated. Specifically, a network pre-trained on the low-fidelity Gaussian wake model is fine-tuned in order to obtain accurate wake results for the mid-fidelity Curl wake model. The overall performance of WakeNet is validated on various wake steering control and layout optimisation scenarios, obtaining at least 90% of the power gained by the FLORIS optimiser. Moreover, the Curl-based WakeNet provides similar power gains to FLORIS, two orders of magnitude faster. These promising results show that generalised wake modelling with ML tools can be accurate enough to contribute towards robust real-time active yaw and layout optimisation under uncertainty, while producing realistic optimised configurations at a fraction of the computational cost.
</p>

<p align="center">
  <img src="https://github.com/soanagno/wakeNet/blob/master/wakenet_fig.png">
</p>

## Paper available at:

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
