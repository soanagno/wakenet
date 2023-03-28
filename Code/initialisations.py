import numpy as np
from packages import json
from packages import torch
import floris.tools as wfct
from floris.tools import static_class as sc


#                              Initialisation of variables                                         #
# =================================================================================================#

# Open JSON file (change based on the wake model)
neural_info = open(
    "example_inputs/inputs_gauss.json",
)


# returns JSON object as a dictionary
data = json.load(neural_info)
# Close JSON file
neural_info.close()

# Turbine parameters
hh = data["turbine"]["cut_plane"]  # hub height
file_path = data["turbine"]["file_path"]

# Data creation parameters
train_net = data["data"]["train_net"]
make_data = data["data"]["make_data"]
save_data = data["data"]["save_data"]
local_ti = data["data"]["local_ti"]
local_pw = data["data"]["local_pw"]
curl = data["data"]["curl"]
weather = data["data"]["weather"]
row_major = data["data"]["row_major"]
x_bounds = data["data"]["x_bounds"]
y_bounds = data["data"]["y_bounds"]
data_size = data["data"]["data_size"]
batch_size = data["data"]["batch_size"]
dimx = data["data"]["dimx"]
dimy = data["data"]["dimy"]
dim1 = data["data"]["dim1"]
dim2 = data["data"]["dim2"]
cubes = data["data"]["cubes"]
norm = data["data"]["norm"]
inputs = data["data"]["inputs"]
plot_curves = data["data"]["plot_curves"]
result_plots = data["data"]["result_plots"]
full_domain = data["data"]["full_domain"]
defo = data["data"]["defo"]

# Data range
ws_range = data["data_range"]["ws_range"]
ti_range = data["data_range"]["ti_range"]
yw_range = data["data_range"]["yw_range"]
hb_range = data["data_range"]["hb_range"]

# Training hyperparameters
# device = data["training"]["device"]
if train_net == True:
    device = "cuda"
else:
    device = "cpu"
parallel = data["training"]["parallel"]
para_workers = data["training"]["para_workers"]
seed = data["training"]["seed"]
epochs = data["training"]["epochs"]
lr = data["training"]["lr"]
momentum = data["training"]["momentum"]
test_batch_size = data["training"]["test_batch_size"]
weight_decay = data["training"]["weight_decay"]
workers = data["training"]["workers"]
train_slice = data["training"]["train_slice"]
val_slice = data["training"]["val_slice"]
test_slice = data["training"]["test_slice"]
opt_method = data["training"]["opt_method"]

# Results parameters
weights_path = data["results"]["weights_path"]
fltr = data["results"]["fltr"]
denoise = data["results"]["denoise"]
contours_on = data["results"]["contours_on"]

# Optimisation boundaries
opt_xbound = data["optimisation"]["opt_xbound"]
opt_ybound = data["optimisation"]["opt_ybound"]
yaw_ini = data["optimisation"]["yaw_ini"]

# Opening turbine JSON file
f = open(
    file_path,
)

# returns JSON object as a dictionary
data2 = json.load(f)
f.close()

# Set GPU if Available
if device == "cuda":
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = "cuda"
    else:
        device = "cpu"
        print("No GPU available! Running on CPU.")

# Get turbine cp curve
cp = np.array(data2["turbine"]["properties"]["power_thrust_table"]["power"])
wind_speed = np.array(
    data2["turbine"]["properties"]["power_thrust_table"]["wind_speed"]
)

# Read turbine json
sc.x_bounds = x_bounds
sc.y_bounds = y_bounds
fi = wfct.floris_interface.FlorisInterface(file_path)
D = fi.floris.farm.turbines[0].rotor_diameter  # turbine rotor diameter
D = float(D)


# Define the size of the partition. if full_domain==Flase, defaults at row or column size.
if full_domain == True:
    out_piece = dimx * dimy
elif cubes == 0:
    out_piece = dim1 * dim2
else:
    if row_major == True:
        out_piece = dimy
    else:
        out_piece = dimx

# Calculates ref_point
# (the list of all points of the domain that the DNN is going to be trained on).
rows = int(dimx * dimy / out_piece)
ref_point_x = np.linspace(0, dimy - 1, dimy)
ref_point_y = np.linspace(0, dimx - 1, dimx)
ref_point = np.zeros((dimx * dimy, 2))
k = 0
for i in range(dimy):
    for j in range(dimx):
        ref_point[k, 0] = ref_point_x[i]
        ref_point[k, 1] = ref_point_y[j]
        k += 1
ref_point = ref_point.astype(np.int)

# Wake boundaries definition
if defo == 1:
    x_bounds = None
    y_bounds = None
else:
    x_bounds = (x_bounds[0], x_bounds[1])
    y_bounds = (y_bounds[0], y_bounds[1])
