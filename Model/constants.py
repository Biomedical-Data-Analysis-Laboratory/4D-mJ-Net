import socket
import os

global M,N,SLICING_PIXELS,verbose,USE_PM,DEBUG,ORIGINAL_SHAPE,TIME_LAST,isISLES,root_path,IMAGE_WIDTH,IMAGE_HEIGH,\
    NUMBER_OF_IMAGE_PER_SECTION,N_CLASSES,LABELS,PIXELVALUES,HOT_ONE_WEIGHTS,TO_CATEG,ALPHA,GAMMA,focal_tversky_loss,\
    PREFIX_IMAGES,DATASET_PREFIX,SUFFIX_IMG,colorbar_coord,suffix_partial_weights,threeD_flag,ONE_TIME_POINT,list_PMS,\
    dataFrameColumns,limitedColumns,ENABLE_WATCHDOG,PID_WATCHDOG_PICKLE_PATH,PID_WATCHDOG_FINISHED_PICKLE_PATH


DATASET_PREFIX = "patient"
SUFFIX_IMG = ".tiff"  # ".png"
colorbar_coord = (129, 435)

suffix_partial_weights = "__"
ONE_TIME_POINT = ""

ENABLE_WATCHDOG = True
PID_WATCHDOG_PICKLE_PATH = os.getcwd()+'/PID_list_{}.obj'.format(socket.gethostname())
PID_WATCHDOG_FINISHED_PICKLE_PATH = os.getcwd()+'/PID_finished_list_{}.obj'.format(socket.gethostname())


################################################################################
# GET & SET FUNCTIONS
################################################################################
def getVerbose():
    return verbose


def setVerbose(v):
    global verbose
    verbose = v


################################################################################
def getDEBUG():
    return DEBUG


def setDEBUG(d):
    global DEBUG
    DEBUG = d
    

################################################################################
def getM():
    return M


def getN():
    return N


def getSLICING_PIXELS():
    return SLICING_PIXELS


def setTileDimension(t):
    global M, N, SLICING_PIXELS
    if t is not None:
        M = int(t)
        N = int(t)
        SLICING_PIXELS = int(M/4)
    else:
        M = 16
        N = 16
        SLICING_PIXELS = int(M/4)


################################################################################
def getIMAGE_WIDTH():
    return IMAGE_WIDTH


def getIMAGE_HEIGHT():
    return IMAGE_HEIGHT
    
    
def setImageDimension(d):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    if d is not None: IMAGE_WIDTH, IMAGE_HEIGHT = int(d), int(d)
    else: IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512


################################################################################
def get3DFlag():
    return threeD_flag


def set3DFlag(flag):
    global threeD_flag
    threeD_flag = ""
    if flag: threeD_flag = "_3D"


################################################################################
def getONETIMEPOINT():
    return ONE_TIME_POINT


def setONETIMEPOINT(timepoint):
    global ONE_TIME_POINT
    ONE_TIME_POINT = "_" + timepoint


################################################################################
def getRootPath():
    return root_path


def setRootPath(path):
    global root_path
    root_path = path


################################################################################
def getPrefixImages():
    return PREFIX_IMAGES


def setPrefix(prefix):
    global PREFIX_IMAGES
    PREFIX_IMAGES = prefix if prefix is not None else "PA"


################################################################################
def getUSE_PM():
    return USE_PM


def getList_PMS():
    return list_PMS


def setUSE_PM(pm):
    global USE_PM, list_PMS
    USE_PM = pm
    if USE_PM and getIsISLES2018(): list_PMS = ["CBF", "CBV", "MTT", "TMAX"]
    else: list_PMS = ["CBF", "CBV", "TTP", "TMAX", "MIP"]


################################################################################
def getIsISLES2018():
    return isISLES


def setISLES2018(isles):
    global isISLES
    isISLES = isles
    if isISLES:
        setImageDimension(256)
        setLimitedColumns(True)


################################################################################
def hasLimitedColumns():
    return limitedColumns


def getDataFrameColumns():
    return dataFrameColumns


def setLimitedColumns(limcols):
    global dataFrameColumns, limitedColumns
    if limcols:
        limitedColumns = True
        dataFrameColumns = ["patient_id", "label", "pixels", "CBF", "CBV", "MTT", "TMAX", "ground_truth", "label_code",
                            "x_y", "data_aug_idx", "timeIndex", "sliceIndex"]
    else:
        limitedColumns = False
        dataFrameColumns = ['patient_id', 'label', 'pixels', 'CBF', 'CBV', 'TTP', 'TMAX', "MIP", "NIHSS", 'ground_truth',
                            'x_y', 'data_aug_idx','timeIndex', 'sliceIndex', 'severity', "age", "gender", 'label_code']


################################################################################
def getTIMELAST():
    return TIME_LAST


def setTimeLast(timelast):
    global TIME_LAST
    TIME_LAST = timelast


################################################################################
def getTO_CATEG():
    return TO_CATEG


def setTO_CATEG(flag):
    global TO_CATEG
    TO_CATEG = flag


################################################################################
def getORIGINAL_SHAPE():
    return ORIGINAL_SHAPE


def getPIXELVALUES():
    return PIXELVALUES


def setOriginalShape(o):
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
def getN_CLASSES():
    return N_CLASSES


def getLABELS():
    return LABELS


def setNumberOfClasses(c):
    global N_CLASSES, LABELS, PIXELVALUES, HOT_ONE_WEIGHTS, GAMMA, ALPHA
    N_CLASSES = c

    if N_CLASSES == 2:
        LABELS = ["background", "core"]
        PIXELVALUES = [0, 255]
        HOT_ONE_WEIGHTS = [[0.1, 100]]
        GAMMA = [[2., 2.]]
        ALPHA = [[0.25,0.25]]
    elif N_CLASSES == 3:
        LABELS = ["background", "penumbra", "core"]
        PIXELVALUES = [0, 170, 255]
        HOT_ONE_WEIGHTS = [[0.1, 50, 440]]
        GAMMA = [[2., 2., 2.]]
        ALPHA = [[0.25, 0.25, 0.25]]
    else:
        LABELS = ["background", "brain", "penumbra", "core"]  # background:0, brain:85, penumbra:170, core:255
        PIXELVALUES = [0, 85, 170, 255]
        HOT_ONE_WEIGHTS = [[0, 0.1, 50, 440]]
        # hyperparameters for the multi focal loss
        ALPHA = [[0.25, 0.25, 0.25, 0.25]]
        GAMMA = [[2., 2., 2., 2.]]


################################################################################
def getFocal_Tversky():
    return focal_tversky_loss


def setFocal_Tversky(hyperparameters):
    global focal_tversky_loss

    focal_tversky_loss = {
        "alpha": 0.7,
        "gamma": 1.33
    }

    for key in hyperparameters.keys(): focal_tversky_loss[key] = hyperparameters[key]


################################################################################
def getWeights():
    return HOT_ONE_WEIGHTS


def setWeights(weights):
    global HOT_ONE_WEIGHTS
    if weights is not None: HOT_ONE_WEIGHTS = [weights]







