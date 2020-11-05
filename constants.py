verbose = False
DEBUG = False
ORIGINAL_SHAPE = False

root_path = ""

M, N = 16, 16
SLICING_PIXELS = int(M/4)
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NUMBER_OF_IMAGE_PER_SECTION = 30  # number of image (divided by time) for each section of the brain
N_CLASSES = 4
LABELS = ["background", "brain", "penumbra", "core"]  # background:0, brain:85, penumbra:170, core:255
PIXELVALUES = [0, 85, 170, 255]
# weights for the categorical cross entropy: 1) position: brain, 2) penumbra, 3) core, 4) background
HOT_ONE_WEIGHTS = [[0.1, 0.2, 1.0, 1.0]]
PREFIX_IMAGES = "PA"
DATASET_PREFIX = "patient"
SUFFIX_IMG = ".tiff"  # ".png"

suffix_partial_weights = "__"
threeD_flag = ""
ONE_TIME_POINT = ""

dataFrameColumns = ['patient_id', 'label', 'pixels', 'ground_truth', 'x_y', 'data_aug_idx', 'timeIndex', 'sliceIndex', 'severity', 'label_code']

################################################################################
def getVerbose():
    return verbose

def getDEBUG():
    return DEBUG

def getM():
    return M

def getN():
    return N

def get3DFlag():
    return threeD_flag

def getONETIMEPOINT():
    return ONE_TIME_POINT

def getRootPath():
    return root_path

def getPrefixImages():
    return PREFIX_IMAGES

################################################################################
################################################################################
# Functions used to set the various GLOBAl variables
def setVerbose(v):
    global verbose
    verbose = v

def setDEBUG(d):
    global DEBUG
    DEBUG = d

def setOriginalShape(o):
    global ORIGINAL_SHAPE, PIXELVALUES
    ORIGINAL_SHAPE = o
    PIXELVALUES = [255, 1, 76, 150]

def setTileDimension(t):
    global M, N, SLICING_PIXELS
    if t is not None:
        M = int(t)
        N = int(t)
        SLICING_PIXELS = int(M/4)

def setImageDimension(d):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    if d is not None:
        IMAGE_WIDTH = int(d)
        IMAGE_HEIGHT = int(d)

def setRootPath(path):
    global root_path
    root_path = path

def setImagePerSection(num):
    global NUMBER_OF_IMAGE_PER_SECTION
    NUMBER_OF_IMAGE_PER_SECTION = num

def setNumberOfClasses(c):
    global N_CLASSES, LABELS, PIXELVALUES, HOT_ONE_WEIGHTS

    if c==2:
        N_CLASSES = c
        LABELS = ["background", "core"]
        PIXELVALUES = [0, 255]
        HOT_ONE_WEIGHTS = [[0.1, 1.0]]
    elif c==3:
        N_CLASSES = c
        LABELS = ["background", "penumbra", "core"]
        PIXELVALUES = [0, 76, 255]
        HOT_ONE_WEIGHTS = [[0.1, 1.0, 1.0]]

def set3DFlag():
    global threeD_flag
    threeD_flag = "_3D"

def setONETIMEPOINT(timepoint):
    global ONE_TIME_POINT
    ONE_TIME_POINT = "_"+timepoint

def setPrefixImagesSUS2020_v2():
    global PREFIX_IMAGES
    PREFIX_IMAGES = "CTP_"
