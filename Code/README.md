Independent Research Project
============================

## Instructions

Simply edit the ```synth``` value in the ```info.json``` to "1" for data creation/training or to "0" for evaluation. Please refer to the ```main_train_test.py``` file for an indicative demonstration. The ```main_scaling.py``` outputs some additional plots used in the Report (for both ```synth```), namely the wake dataset indicative samples and the computational time vs number of turbines scaling plot.

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

* FLORIS (2.1.1)
* Numpy (1.17.2)
* SciPy (1.3.1)
* Matplotlib (3.1.1)
* Torch (1.5.1)
* Dask (2.9.0), optional
* Cuda (10.1), optional

