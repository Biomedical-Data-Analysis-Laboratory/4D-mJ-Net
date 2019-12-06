#global verbose, root_path
verbose = False
root_path = ""

# just for testing and debugging 
DEBUG = False

default_setting_filename = "defaults_settings.json"

M, N = 16, 16
SLICING_PIXELS = int(M/4)
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
NUMBER_OF_IMAGE_PER_SECTION = 30 # number of image (divided by time) for each section of the brain
SAMPLES = 50
LABELS = ["background", "brain", "penumbra", "core"] # background:255, brain:0, penumbra:~76, core:~150
PREFIX_IMAGES = "PA"



def getVerbose():
    return verbose

def getRootPath():
    return root_path

def setVerbose(v):
    global verbose
    verbose = v

def setRootPath(path):
    global root_path
    root_path = path
