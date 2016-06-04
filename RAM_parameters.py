
save_dir = "save-3scales/"
save_prefix = "save"
start_step = 10000
load_path = None
# load_path = save_dir + "save.ckpt"

minRadius = 3 # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8 # fixed resolution of sensor
sensorArea = sensorBandwidth**2
depth = 3 # zooms
channels = 1 # grayscale
totalSensorBandwidth = depth * sensorBandwidth * sensorBandwidth * channels
batch_size = 20

hg_size = 128
hl_size = 128

# g_size = 256
g_size = 128
cell_size = 256
cell_out_size = cell_size

glimpses = 5
n_classes = 10

lr = 1e-3
max_iters = 1000000

mnist_size = 28

loc_sd = 0.1

lmda = 2. #cost weights
