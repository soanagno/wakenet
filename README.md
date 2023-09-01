# WakeNet: Deep Neural Networks for wind farm wake modelling and optimisation

## Abstract

<p align="justify">
Wind farm modelling is an area of rapidly increasing interest with numerous analytical and computational-based approaches developed to extend the margins of wind farm efficiency and maximise power production. In this work, we present the novel ML framework WakeNet, which reproduces generalised 2D turbine wake velocity fields at hub-height, with a mean accuracy of 99.8% compared to the solution calculated by the state-of-the-art wind farm modelling software FLORIS. As the generation of sufficient high-fidelity data for network training purposes can be cost-prohibitive, the utility of multi-fidelity transfer learning has also been investigated. Specifically, a network pre-trained on the low-fidelity Gaussian wake model is fine-tuned in order to obtain accurate wake results for the mid-fidelity Curl wake model. The overall performance of WakeNet is validated on various wake steering control and layout optimisation scenarios, obtaining at least 90% of the power gained by the FLORIS optimiser. Moreover, the Curl-based WakeNet provides similar power gains to FLORIS, two orders of magnitude faster. These promising results show that generalised wake modelling with ML tools can be accurate enough to contribute towards robust real-time active yaw and layout optimisation under uncertainty, while producing realistic optimised configurations at a fraction of the computational cost.
</p>

<p align="center">
  <img src="https://github.com/soanagno/wakeNet/blob/master/wakenet_fig.png">
</p>

## Paper available at:

https://arxiv.org/pdf/2303.16274.pdf

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

## Instructions

1) Edit the ```.json``` input file:

Simply set the ```train_net``` value to "true" for training mode or to "false" for evaluation mode. On training mode, set ```make_data``` to "true" if you want to generate the wakes or to "false" if you want to read the wakes from the "wake_dataset" folder within the same directory. To create the "wake_dataset" folder set ```save_data``` to true and run the ```example_main.py```.

2) To either train or evaluate execute the following command:

```python -u example_main.py```

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
