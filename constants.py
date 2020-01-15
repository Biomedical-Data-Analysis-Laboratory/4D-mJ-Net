verbose = False
root_path = ""

default_setting_filename = "Settings/defaults_settings.json"

M, N = 16, 16
SLICING_PIXELS = int(M/4)
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NUMBER_OF_IMAGE_PER_SECTION = 30 # number of image (divided by time) for each section of the brain
SAMPLES = 50
LABELS = ["background", "brain", "penumbra", "core"] # background:255, brain:0, penumbra:~76, core:~150
PIXELVALUES = [255, 0, 76, 150]
PREFIX_IMAGES = "PA"
suffix_partial_weights = "__"

dataFrameColumnsTest = ['patient_id', 'label', 'pixels', 'ground_truth'] # without the "label_code"
dataFrameColumns = ['patient_id', 'label', 'pixels', 'ground_truth', "label_code"]


def getVerbose():
    return verbose

def getDEBUG():
    return DEBUG

def getRootPath():
    return root_path

def setVerbose(v):
    global verbose
    verbose = v

def setDEBUG(d):
    global DEBUG
    DEBUG = d

def setRootPath(path):
    global root_path
    root_path = path
