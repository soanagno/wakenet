wakeNet
============================

## Instructions

1) To either train or evaluate execute the following command:

```python -u train_test_dnn.py```

2) Make sure to edit the ```inputs.json``` file accordingly:

Simply set the ```train_net``` value to "1" for training mode or to "0" for evaluation mode. On training mode, set ```make_data``` to "true" if you want to generate the wakes or to "false" if you want to read the wakes from the "wake_dataset" within the same directory. To create the "wake_dataset" folder set ```save_data``` to true and run the ```main_train-test.py``` once. The ```turbine_scaling.py``` outputs some additional plots used in the Report (for both ```train_net``` values), namely the wake dataset indicative samples and the computational time vs number of turbines scaling plot. Note that the example .json files in "example_inputs" have to be renamed as "inputs.json" to be used.

More specifically, each setting parameter included in ```inputs.json``` and its functionality is listed bellow as it appears in the json:

#### turbine
```
file_path (string): initialisation file path for FLORIS
cut_plane (float): height of the 2D plane of the DNN training. default is hub height.
```
#### data
```
train_net (boolean): 0 for training mode, 1 for evaluation mode.
make_data (boolean): true for wake generation, false to load wakes from wake_dataset folder.
save_data (boolean): set to true to create wake_dataset folder.
curl (boolean): set to true to train dnn on the curl model and false on the gaussian model (experimental).
weather (boolean): create dataset based on realistic wind data (data_size and batch_size must be changed).
data_size (float): total number of wakes used for training
batch_size (float): batch size of each backward propagation.
x_bnds (vector): x-boundaries of wakes produced by FLORIS.
y_bnds (vector): y-boundaries of wakes produced by FLORIS.
full_domain (boolean): 1 for use of full domain during, 0 for domain partitioning (using sub-networks).
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
device (cpu or cuda): defines the computation device.
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

