import os

LABEL_MAPPING = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
LABEL_MAPPING_MMS = {"Other": 0, "NOR": 1, "DCM": 2,"HCM": 3} # Removi AHS pois há apenas um caso valido no dataset"HHD": 1,"ARV": 4, 
NUM_CLASSES_MMS = 7
TARGET_SHAPE = (192, 192, 10)
NUM_CLASSES = 5
MAX_TIME_DIM = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(BASE_DIR, './Datasets')
ACDC_TRAINING_PATH = os.path.join(DATASETS_PATH, './ACDC/database/training/')
ACDC_TESTING_PATH = os.path.join(DATASETS_PATH, './ACDC/database/testing/')
OUTPUT_PATH = os.path.join(BASE_DIR, './output/')
WEIGHT_PATH = os.path.join(BASE_DIR, './weights/')
CSV_PATH=os.path.join(DATASETS_PATH,'./OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
MMs_PATH = os.path.join(DATASETS_PATH,'./OpenDataset/')
MMs_REESPACADO = os.path.join(DATASETS_PATH,'./mms_resampled/')
MMs_VALIDATION = os.path.join(DATASETS_PATH,'./OpenDataset/Validation/')
ZOOM=1.0
GRID=(6,6)
CLIP=2.25
KAGGLE_PATH = os.path.join(BASE_DIR,'./kaggle/input/')