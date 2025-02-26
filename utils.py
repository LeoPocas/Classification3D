import os

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
TARGET_SHAPE = (96, 96, 16)
NUM_CLASSES = 5
MAX_TIME_DIM = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACDC_TRAINING_PATH = os.path.join(BASE_DIR, './ACDC/database/training/')
ACDC_TESTING_PATH = os.path.join(BASE_DIR, './ACDC/database/testing/')
OUTPUT_PATH = os.path.join(BASE_DIR, './output/')
WEIGHT_PATH = os.path.join(BASE_DIR, './weights/')
ZOOM=1.4
GRID=(2,2)
CLIP=0.5