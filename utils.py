import os

import numpy as np

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
MMs_PATH = os.path.join(DATASETS_PATH,'./OpenDataset/')
MMs_REESPACADO = os.path.join(DATASETS_PATH,'./mms_resampled/')
CSV_PATH=os.path.join(MMs_REESPACADO,'./211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
MMs_VALIDATION = os.path.join(DATASETS_PATH,'./OpenDataset/Validation/')
INCOR_PATH = os.path.join(DATASETS_PATH,'./dataset incor/')
INCOR_RESAMPLED_TRAINING_PATH = os.path.join(DATASETS_PATH,'./Incor/Training/')
INCOR_RESAMPLED_TESTING_PATH = os.path.join(DATASETS_PATH,'./Incor/Testing/')
INCOR_RESAMPLED_PATH = os.path.join(DATASETS_PATH,'./Incor/')
ROI_PATH = os.path.join(OUTPUT_PATH, './ROI_locations.txt')
ZOOM=0.7
GRID=(16,16)
CLIP=0.5
SPACING = (1.0, 1.0, 10)


def select_training_patients(patient_labels, fraction, random_seed):
    """Select a deterministic, label-stratified subset of training patients."""
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("training_data_fraction deve estar entre 0 e 1.")

    patient_labels = list(patient_labels.items())
    if fraction == 1 or not patient_labels:
        return {patient_id for patient_id, _ in patient_labels}

    rng = np.random.default_rng(random_seed)
    selected = set()
    labels = sorted({label for _, label in patient_labels})
    for label in labels:
        class_patients = sorted(
            patient_id for patient_id, patient_label in patient_labels
            if patient_label == label
        )
        sample_size = max(1, int(np.ceil(len(class_patients) * fraction)))
        selected.update(
            class_patients[index]
            for index in rng.choice(len(class_patients), size=sample_size, replace=False)
        )
    return selected
KAGGLE_PATH = os.path.join(BASE_DIR,'./kaggle/input/')
LOGS_DIR = os.path.join(BASE_DIR, "tempos")


def use_early_stopping(config):
    """Return whether the configured training uses early stopping."""
    if "early_stopping" in config:
        return bool(config["early_stopping"])

    dataset = config.get("dataset", "incor").lower()
    if dataset == "mms":
        return True

    model_mode = config.get("model_mode", "concat")
    training_fraction = float(config.get("training_data_fraction", 1.0))
    return (
        dataset == "incor"
        and model_mode in {"concat", "concat_volume", "early_channel"}
        and training_fraction < 1
    )