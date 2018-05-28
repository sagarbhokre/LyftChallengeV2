BG_ID = 0
CAR_ID = 1
ROAD_ID = 2
n_classes = 3

input_height = 600
input_width = 800
image_shape = (input_height, input_width)

nw_height = 304
nw_width = 800
nw_shape = (nw_height, nw_width)

OFFSET_HIGH=194
OFFSET_LOW=OFFSET_HIGH+nw_height

visualize = True
enable_profiling = False

model_path = 'checkpoint/ep-009-val_loss-1.0074.hdf5'
