import os

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
LABEL_MAPPING_MMS = {"Other": 0, "NOR": 1, "DCM": 2,"HCM": 3} # Removi AHS pois há apenas um caso valido no dataset"HHD": 1,"ARV": 4, 
NUM_CLASSES_MMS = 7
TARGET_SHAPE = (128, 128, 16)
NUM_CLASSES = 5
MAX_TIME_DIM = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACDC_TRAINING_PATH = os.path.join(BASE_DIR, './ACDC/database/training/')
ACDC_TESTING_PATH = os.path.join(BASE_DIR, './ACDC/database/testing/')
OUTPUT_PATH = os.path.join(BASE_DIR, './output/')
WEIGHT_PATH = os.path.join(BASE_DIR, './weights/')
CSV_PATH=os.path.join(BASE_DIR,'./OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
MMs_PATH = os.path.join(BASE_DIR,'./OpenDataset/')
MMs_VALIDATION = os.path.join(BASE_DIR,'./OpenDataset/Validation/')
ZOOM=1.0
GRID=(1,1)
CLIP=0.2
KAGGLE_PATH = os.path.join(BASE_DIR,'./kaggle/input/')