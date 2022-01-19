# wakeNet: Deep Neural Networks for Wind Turbine Wake modelling and Optimisation

## Abstract

Offshore wind farm modelling has been an area of increasing interest over the last two decades, with numerous remarkable analytical as well as computational approaches attempting to accurately describe the complex wind flows taking place. The ultimate aim is the development of tools that can be used to provide designs that increase the efficiency in power production. This work presents an open-source Machine Learning (ML) framework for the rapid modelling of wind farm flow field, using a Deep Feed Forward (DFF) neural network architecture, trained on approximate turbine wake fields, as calculated by the state-of-the-art wind farm modelling software FLORIS. The constructed neural model is capable of accurately reproducing the single wake deficits on the hub level of a 5MW wind turbine under yaw and a wide range of inlet hub speed and turbulence intensity conditions, at least an order of magnitude faster than the analytical wake based solution method, yielding results of 98.5% mean accuracy. A superposition algorithm is also developed to construct the wind farm domain of superimposed individual wakes. The ability of the trained neural network in providing reliable results is further demonstrated by performing wind farm yaw and layout optimisations, where the DFF produces optimal solutions one order of magnitude faster than the same optimisation carried out by FLORIS, across a wide range of wind conditions. A promising advantage of the present approach is that its performance and accuracy is expected to increase even further when trained on high-fidelity CFD data through transfer learning, while its computational cost for evaluation is kept low and constant.

<p align="center">
  <img src="https://github.com/soanagno/wakeNet/blob/master/dnn_fig.png">
</p>


## Instructions

Simply edit the ```synth``` value in the ```info.json``` to "1" for synthesizing wakes and network training or to "0" for evaluation. Please refer to the ```train_test_dnn.py``` file for an indicative demonstration. The ```turbine_scaling.py``` outputs some additional plots used in the Report (for both ```synth``` values), namely the wake dataset indicative samples and the computational time vs number of turbines scaling plot.

More specifically, each setting parameter included in ```info.json``` and its functionality is listed bellow as it appears in the json:

#### turbine
```
file_path (string): initialisation file path for FLORIS
cut_plane (float): height of the 2D plane of the DNN training. default is hub height.
```
#### data
```
synth (boolean): 0 for data synthesizing and training, 1 for evaluation mode.
data_size (float): total number of wakes used for training
batch_size (float): batch size of each backward propagation.
x_bnds (vector): x-boundaries of wakes produced by FLORIS.
y_bnds (vector): y-boundaries of wakes produced by FLORIS.
full_domain (boolean): 0 for domain partitioning (using sub-networks), 1 for use of full domain during training.
row_major (boolean): row-major (1) or column major (0) partitioning.
dimx (float): x-resolution of wake domain as calculated by FLORIS.
dimy (float): y-resolution of wake domain as calculated by FLORIS.
cubes (boolean): if true, block-wise partitioning.
dim1 (float): x-resolution of each block.
dim2 (float): y-resolution of each block. Note that dimx*dimy must be divisible by dim1*dim2.
norm (1, 2 or 3): mode of normalisation.
inputs (1, 2 or 3): number of inputs for the neural network training (ws, ti, yw).
plot_curves (boolean): if true, plot validation and training loss/accuracy curves.
result_plots (boolean): if true, after training plot the evaluation of each wake.
defo (boolean): use default FLORIS dimensions of single wake.
```
#### data range
```
ws_range (vector): inlet wind speed range.
ti_range (vector): wind turbulence intensity range.
yw_range (vector): turbine yaw angle range.
hb_range (vector): hub slice range (unused).
```
#### training
```
device (cpu or gpu): defines the computation device.
parallel (boolean): if true and also device="cpu", initialises parallel computations.
para_workers (int): number of parallel workers.
seed (int): random seed for gpu computations.
epochs (int): training epochs.
lr (float): learning rate.
momentum (float): learning momentum (only for SGD).
opt_method (Rprop or SGD): switches between Rprop and SGD optimisers.
test_batch_size (float, optional): sets the test batch size.
weight_decay (float): learning weight decay (only for SGD).
workers (int, optional): number of workers carrying the training batches.
train_slice (float): percentage of training set.
val_slice (float): percentage of validation set.
test_slice (float, optional): percentage of test set.
```
#### results
```
weights_path (string): path of trained weights.
fltr (float, optional): filter for wake tracking.
denoise (int, optional): denoises wake/eliminates scattering on wake edges.
```
#### optimisation
```
yaw_ini (float): initial yaw of turbines to be optimised.
opt_cbound (float): x-boundary optimisation contraint defined in number of diameters (D).
opt_cbound (float): y-boundary optimisation contraint defined in number of diameters (D).
```

Note that the user only needs to change the synth value depending on the use, as the default values were the ones used in the Report.


## Requirements

* Python (3.9.7)
* FLORIS (2.4)
* Numpy (1.21.3)
* SciPy (1.7.1)
* Matplotlib (3.4.3)
* Torch (1.10.0)
* Dask (2021.10.0), optional
* Cuda (11.3), optional


