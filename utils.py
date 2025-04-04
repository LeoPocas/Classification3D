import os

LABEL_MAPPING = {'NOR': 0, 'DCM': 1, 'HCM': 2}
LABEL_MAPPING_MMS = {"NOR": 0, "DCM": 1,"HCM": 2} # Removi AHS pois há apenas um caso valido no dataset"HHD": 1,"ARV": 4, "Other": 0, 
NUM_CLASSES_MMS = 3
TARGET_SHAPE = (192, 192, 12)
NUM_CLASSES = 3
MAX_TIME_DIM = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(BASE_DIR, './Datasets')
ACDC_TRAINING_PATH = os.path.join(DATASETS_PATH, './ACDC/database/training/')
ACDC_TESTING_PATH = os.path.join(DATASETS_PATH, './ACDC/database/testing/')
ACDC_REESPACADO = os.path.join(DATASETS_PATH, './acdc_resampled/')
ACDC_REESPACADO_TRAINING = os.path.join(ACDC_REESPACADO, './training/')
ACDC_REESPACADO_TESTING = os.path.join(ACDC_REESPACADO, './testing/')
OUTPUT_PATH = os.path.join(BASE_DIR, './output/')
WEIGHT_PATH = os.path.join(BASE_DIR, './weights/')
CSV_PATH=os.path.join(DATASETS_PATH,'./OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
MMs_PATH = os.path.join(DATASETS_PATH,'./OpenDataset/')
MMs_REESPACADO = os.path.join(DATASETS_PATH,'./mms_resampled/')
MMs_VALIDATION = os.path.join(DATASETS_PATH,'./OpenDataset/Validation/')
INCOR_PATH = os.path.join(DATASETS_PATH,'./dataset incor/Dilatado/')
ROI_PATH = os.path.join(BASE_DIR, './ROI_locations.txt')
ZOOM=1.0
GRID=(6,6)
CLIP=2.25
KAGGLE_PATH = os.path.join(BASE_DIR,'./kaggle/input/')