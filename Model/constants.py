import socket
import os

global M,N,SLICING_PIXELS,verbose,USE_PM,DEBUG,ORIGINAL_SHAPE,TIME_LAST,isISLES,root_path,IMAGE_WIDTH,IMAGE_HEIGHT,\
    NUMBER_OF_IMAGE_PER_SECTION,N_CLASSES,LABELS,PIXELVALUES,HOT_ONE_WEIGHTS,TO_CATEG,ALPHA,GAMMA,focal_tversky_loss,\
    PREFIX_IMAGES,DATASET_PREFIX,SUFFIX_IMG,colorbar_coord,suffix_partial_weights,threeD_flag,ONE_TIME_POINT,list_PMS,\
    DF_columns,limited_columns,ENABLE_WATCHDOG,PID_WATCHDOG_PICKLE_PATH,PID_WATCHDOG_FINISHED_PICKLE_PATH, TO_FLAT


DATASET_PREFIX = "patient"
SUFFIX_IMG = ".tiff"  # ".png"
colorbar_coord = (129, 435)

suffix_partial_weights = "__"
ONE_TIME_POINT = ""
focal_tversky_loss = {"alpha": 0.7, "gamma": 1.33}

ENABLE_WATCHDOG = True
PID_WATCHDOG_PICKLE_PATH = os.getcwd()+'/PID_list_{}.obj'.format(socket.gethostname())
PID_WATCHDOG_FINISHED_PICKLE_PATH = os.getcwd()+'/PID_finished_list_{}.obj'.format(socket.gethostname())


################################################################################
# GET & SET FUNCTIONS
################################################################################
def is_verbose():
    return verbose


def set_verbose(v):
    global verbose
    verbose = v


################################################################################
def is_debug():
    return DEBUG


def set_debug(d):
    global DEBUG
    DEBUG = d


################################################################################
def is_to_flat():
    return TO_FLAT


def set_to_flat(f):
    global TO_FLAT
    TO_FLAT = f


################################################################################
def get_m():
    return M


def get_n():
    return N


def get_slice_pixels():
    return SLICING_PIXELS


def set_tile_dim(t):
    global M, N, SLICING_PIXELS
    if t is not None:
        M = int(t)
        N = int(t)
        SLICING_PIXELS = int(M/4)
    else:
        M = 512
        N = 512
        SLICING_PIXELS = int(M/4)


################################################################################
def get_img_width():
    return IMAGE_WIDTH


def get_img_height():
    return IMAGE_HEIGHT
    
    
def set_img_dim(d):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    if d is not None: IMAGE_WIDTH, IMAGE_HEIGHT = int(d), int(d)
    else: IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512


################################################################################
def is_3D():
    return threeD_flag


def set_3D_flag(flag):
    global threeD_flag
    threeD_flag = ""
    if flag: threeD_flag = "_3D"


################################################################################
def get_onetimepoint():
    return ONE_TIME_POINT


def set_onetimepoint(timepoint):
    global ONE_TIME_POINT
    ONE_TIME_POINT = "_" + timepoint


################################################################################
def get_rootpath():
    return root_path


def set_rootpath(path):
    global root_path
    root_path = path


################################################################################
def get_prefix_img():
    return PREFIX_IMAGES


def set_prefix(prefix):
    global PREFIX_IMAGES
    PREFIX_IMAGES = prefix if prefix is not None else "PA"


################################################################################
def get_USE_PM():
    return USE_PM


def get_list_PMS():
    return list_PMS


def set_USE_PM(pm):
    global USE_PM, list_PMS
    USE_PM = pm
    if USE_PM and is_ISLES2018(): list_PMS = ["CBF", "CBV", "MTT", "TMAX"]
    else: list_PMS = ["CBF", "CBV", "TTP", "TMAX", "MIP"]


################################################################################
def is_ISLES2018():
    return isISLES


def set_ISLES2018(isles):
    global isISLES
    isISLES = isles
    if isISLES:
        set_img_dim(256)
        set_limited_columns(True)


################################################################################
def has_limited_columns():
    return limited_columns


def get_DF_columns():
    return DF_columns


def set_limited_columns(limcols):
    global DF_columns, limited_columns
    if limcols:
        limited_columns = True
        DF_columns = ["patient_id", "label", "pixels", "CBF", "CBV", "MTT", "TMAX", "ground_truth", "label_code",
                      "x_y", "data_aug_idx", "timeIndex", "sliceIndex"]
    else:
        limited_columns = False
        DF_columns = ['patient_id', 'label', 'pixels', 'CBF', 'CBV', 'TTP', 'TMAX', "MIP", "NIHSS", 'ground_truth',
                      'x_y', 'data_aug_idx','timeIndex', 'sliceIndex', 'severity', "age", "gender", 'label_code']


################################################################################
def is_timelast():
    return TIME_LAST


def set_timelast(timelast):
    global TIME_LAST
    TIME_LAST = timelast


################################################################################
def is_TO_CATEG():
    return TO_CATEG


def set_TO_CATEG(flag):
    global TO_CATEG
    TO_CATEG = flag


################################################################################
def is_orig_shape():
    return ORIGINAL_SHAPE


def get_pixel_values():
    return PIXELVALUES


def set_orig_shape(o):
    global ORIGINAL_SHAPE, PIXELVALUES
    ORIGINAL_SHAPE = o
    PIXELVALUES = [255, 1, 76, 150] if ORIGINAL_SHAPE else [0, 85, 170, 255]


################################################################################
def getNUMBER_OF_IMAGE_PER_SECTION():
    return NUMBER_OF_IMAGE_PER_SECTION


def setImagePerSection(num):
    global NUMBER_OF_IMAGE_PER_SECTION
    NUMBER_OF_IMAGE_PER_SECTION = num   # number of image (divided by time) for each section of the brain


################################################################################
def get_n_classes():
    return N_CLASSES


def get_labels():
    return LABELS


def set_classes(c):
    global N_CLASSES, LABELS, PIXELVALUES, HOT_ONE_WEIGHTS, GAMMA, ALPHA
    N_CLASSES = c

    if N_CLASSES == 2:
        LABELS = ["background", "core"]
        PIXELVALUES = [0, 255]
        HOT_ONE_WEIGHTS = {0:1, 1:1}  # [[0.1, 100]]
        GAMMA = [[2., 2.]]
        ALPHA = [[0.25,0.25]]
    elif N_CLASSES == 3:
        LABELS = ["background", "penumbra", "core"]
        PIXELVALUES = [0, 170, 255]
        HOT_ONE_WEIGHTS = {0:1, 1:1, 2:1}  # [[0.1, 50, 440]]
        GAMMA = [[2., 2., 2.]]
        ALPHA = [[0.25, 0.25, 0.25]]
    else:
        LABELS = ["background", "brain", "penumbra", "core"]  # background:0, brain:85, penumbra:170, core:255
        PIXELVALUES = [0, 85, 170, 255]
        HOT_ONE_WEIGHTS = {0:1, 1:1, 2:1, 3:1}   # [[0, 0.1, 50, 440]]
        # hyperparameters for the multi focal loss
        ALPHA = [[0.25, 0.25, 0.25, 0.25]]
        GAMMA = [[2., 2., 2., 2.]]


################################################################################
def get_Focal_Tversky():
    return focal_tversky_loss


def set_Focal_Tversky(hyperparameters):
    global focal_tversky_loss
    focal_tversky_loss = {"alpha": 0.7, "gamma": 1.33}
    for key in hyperparameters.keys(): focal_tversky_loss[key] = hyperparameters[key]


################################################################################
def get_class_weights_const():
    sorted_keys = sorted(HOT_ONE_WEIGHTS.keys())
    out = []
    for k in sorted_keys: out.append(HOT_ONE_WEIGHTS[k])
    return [out]


def set_class_weights(weights):
    global HOT_ONE_WEIGHTS
    if weights is not None:
        sorted_keys = sorted(HOT_ONE_WEIGHTS.keys())
        for i,w in enumerate(weights): HOT_ONE_WEIGHTS[sorted_keys[i]] = w
