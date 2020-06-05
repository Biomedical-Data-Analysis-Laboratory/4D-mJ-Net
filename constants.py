verbose = False
root_path = ""

default_setting_filename = "Settings/defaults_settings.json"

M, N = 16, 16
SLICING_PIXELS = int(M/4)
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NUMBER_OF_IMAGE_PER_SECTION = 30 # number of image (divided by time) for each section of the brain
SAMPLES = 500
N_CLASSES = 4
LABELS = ["background", "brain", "penumbra", "core"] # background:255, brain:0, penumbra:~76, core:~150
PIXELVALUES = [255, 1, 76, 150]
# weights for the categorical cross entropy: 1) position: brain, 2) penumbra, 3) core, 4) background
HOT_ONE_WEIGHTS = [[0.2, 1.0, 1.0, 0.1]]
PREFIX_IMAGES = "PA"
suffix_partial_weights = "__"

dataFrameColumnsTest = ['patient_id', 'label', 'pixels', 'ground_truth'] # without the "label_code"
dataFrameColumns = ['patient_id', 'label', 'pixels', 'ground_truth', "label_code"]


def getVerbose():
    return verbose

def getDEBUG():
    return DEBUG

def getM():
    return M

def getN():
    return N

def getRootPath():
    return root_path

def setVerbose(v):
    global verbose
    verbose = v

def setDEBUG(d):
    global DEBUG
    DEBUG = d

################################################################################
################################################################################
# Functions used to set the various GLOBAl variables
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

    if c==2: #
        N_CLASSES = c
        LABELS = ["background", "core"]
        PIXELVALUES = [0, 255]
        HOT_ONE_WEIGHTS = [[0.1, 1.0]]
