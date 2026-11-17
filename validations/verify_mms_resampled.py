import argparse
import csv
import os
from collections import defaultdict

import nibabel as nib
import numpy as np


ALLOWED_PATHOLOGIES = {"HCM", "DCM", "NOR"}
DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "Datasets", "mms_resampled"
)
DEFAULT_CSV_PATH = os.path.join(
    DEFAULT_DATASET_DIR, "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
)


def load_patient_info(csv_path):
    patient_info = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            patient_id = row.get("External code", "").strip()
            pathology = row.get("Pathology", "").strip().upper()
            if patient_id:
                patient_info[patient_id] = {
                    "pathology": pathology,
                    "ed": int(row["ED"]) if row.get("ED") else None,
                    "es": int(row["ES"]) if row.get("ES") else None,
                }
    return patient_info


def list_patient_directories(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        entry
        for entry in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, entry))
    )


def inspect_patient_files(patient_path, patient_info):
    report = {
        "nifti_files": 0,
        "invalid_files": [],
        "shapes": [],
        "dtypes": [],
        "scales": [],
        "value_ranges": [],
        "phase_errors": [],
    }
    ed = patient_info.get("ed")
    es = patient_info.get("es")

    for filename in sorted(os.listdir(patient_path)):
        if not filename.endswith((".nii", ".nii.gz")) or "gt" in filename.lower():
            continue

        report["nifti_files"] += 1
        file_path = os.path.join(patient_path, filename)
        try:
            image = nib.load(file_path)
            shape = tuple(image.shape)
            header = image.header
            slope = header.get("scl_slope")
            intercept = header.get("scl_inter")
            report["shapes"].append((filename, shape))
            report["dtypes"].append((filename, str(header.get_data_dtype())))
            report["scales"].append((filename, slope, intercept))
            data = image.get_fdata(dtype=np.float32)
            report["value_ranges"].append(
                (filename, float(np.min(data)), float(np.max(data)))
            )

            if len(shape) != 4:
                report["invalid_files"].append(
                    f"{filename}: shape {shape}, esperado NIfTI 4D"
                )
                continue

            frame_count = shape[3]
            for phase_name, phase in (("ED", ed), ("ES", es)):
                if phase is None:
                    report["phase_errors"].append(
                        f"{filename}: {phase_name} ausente"
                    )
                elif phase < 0 or phase >= frame_count:
                    report["phase_errors"].append(
                        f"{filename}: {phase_name}={phase} fora de [0, {frame_count - 1}]"
                    )
        except Exception as error:
            report["invalid_files"].append(f"{filename}: {error}")

    return report


def find_split_path(dataset_dir, split_name):
    for entry in os.listdir(dataset_dir):
        entry_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(entry_path) and entry.casefold() == split_name.casefold():
            return entry_path
    return os.path.join(dataset_dir, split_name)


def verify_dataset(dataset_dir, csv_path):
    patient_info = load_patient_info(csv_path)
    patients_by_split = {}
    pathology_by_split = defaultdict(list)
    file_reports = []

    for split in ("training", "testing"):
        split_path = find_split_path(dataset_dir, split)
        patient_ids = list_patient_directories(split_path)
        patients_by_split[split] = patient_ids

        for patient_id in patient_ids:
            patient = patient_info.get(patient_id)
            if patient is None:
                pathology_by_split["UNKNOWN_ID"].append(f"{split}/{patient_id}")
            elif patient["pathology"] not in ALLOWED_PATHOLOGIES:
                pathology_by_split[patient["pathology"]].append(f"{split}/{patient_id}")
            else:
                pathology_by_split[patient["pathology"]].append(f"{split}/{patient_id}")

            if patient is not None and patient["pathology"] in ALLOWED_PATHOLOGIES:
                report = inspect_patient_files(
                    os.path.join(split_path, patient_id), patient
                )
                file_reports.append((split, patient_id, report))

    all_dataset_patients = {
        patient_id
        for patient_ids in patients_by_split.values()
        for patient_id in patient_ids
    }
    eligible_csv_patients = {
        patient_id
        for patient_id, patient in patient_info.items()
        if patient["pathology"] in ALLOWED_PATHOLOGIES
    }
    missing_from_dataset = sorted(eligible_csv_patients - all_dataset_patients)

    duplicate_splits = sorted(
        set(patients_by_split["training"]) & set(patients_by_split["testing"])
    )
    invalid_entries = []
    valid_entries = []

    for split, patient_ids in patients_by_split.items():
        for patient_id in patient_ids:
            patient = patient_info.get(patient_id)
            entry = f"{split}/{patient_id}"
            if patient is not None and patient["pathology"] in ALLOWED_PATHOLOGIES:
                valid_entries.append((entry, patient["pathology"]))
            else:
                invalid_entries.append(
                    (entry, patient["pathology"] if patient else "UNKNOWN_ID")
                )

    print(f"Dataset: {os.path.abspath(dataset_dir)}")
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"Patologias permitidas: {', '.join(sorted(ALLOWED_PATHOLOGIES))}")
    print()

    for split in ("training", "testing"):
        print(f"{split}: {len(patients_by_split[split])} pacientes")

    print(f"Total de pacientes unicos nas pastas: {len(all_dataset_patients)}")
    print(f"Pacientes validos encontrados: {len(valid_entries)}")
    print()

    print("Auditoria dos exames:")
    total_files = sum(report["nifti_files"] for _, _, report in file_reports)
    print(f"  Arquivos NIfTI validos lidos: {total_files}")

    invalid_files = [
        (split, patient_id, item)
        for split, patient_id, report in file_reports
        for item in report["invalid_files"]
    ]
    phase_errors = [
        (split, patient_id, item)
        for split, patient_id, report in file_reports
        for item in report["phase_errors"]
    ]
    print(f"  Arquivos NIfTI com erro: {len(invalid_files)}")
    print(f"  Erros de ED/ES: {len(phase_errors)}")

    dtypes = sorted({dtype for _, _, report in file_reports for _, dtype in report["dtypes"]})
    scales = sorted(
        {
            (str(slope), str(intercept))
            for _, _, report in file_reports
            for _, slope, intercept in report["scales"]
        }
    )
    shapes = sorted({shape for _, _, report in file_reports for _, shape in report["shapes"]})
    value_ranges = [
        (split, patient_id, filename, minimum, maximum)
        for split, patient_id, report in file_reports
        for filename, minimum, maximum in report["value_ranges"]
    ]
    print(f"  Dtypes encontrados: {dtypes or 'nenhum'}")
    print(f"  Shapes encontrados: {shapes or 'nenhum'}")
    print(f"  Escalas NIfTI encontradas (slope, intercept): {scales or 'nenhuma'}")
    if value_ranges:
        global_min = min(item[3] for item in value_ranges)
        global_max = max(item[4] for item in value_ranges)
        print(f"  Faixa global dos voxels: [{global_min:.3f}, {global_max:.3f}]")

    if invalid_files:
        print("\nERRO: arquivos NIfTI inválidos:")
        for split, patient_id, error in invalid_files:
            print(f"  {split}/{patient_id}: {error}")

    if phase_errors:
        print("\nERRO: fases ED/ES inválidas:")
        for split, patient_id, error in phase_errors:
            print(f"  {split}/{patient_id}: {error}")

    print("Contagem por patologia encontrada:")
    for pathology in sorted(ALLOWED_PATHOLOGIES):
        count = sum(1 for _, entry_pathology in valid_entries if entry_pathology == pathology)
        print(f"  {pathology}: {count}")
    print()

    if invalid_entries:
        print("ERRO: pacientes desconhecidos ou com patologias nao permitidas:")
        for entry, pathology in invalid_entries:
            print(f"  {entry}: {pathology}")
    else:
        print("OK: todos os pacientes nas pastas possuem HCM, DCM ou NOR no CSV.")

    if duplicate_splits:
        print("\nERRO: pacientes presentes simultaneamente em training e testing:")
        for patient_id in duplicate_splits:
            print(f"  {patient_id}")

    if missing_from_dataset:
        print(
            f"\nPacientes HCM/DCM/NOR do CSV ausentes das duas pastas "
            f"({len(missing_from_dataset)}):"
        )
        for patient_id in missing_from_dataset:
            print(f"  {patient_id}: {patient_info[patient_id]['pathology']}")

    is_valid = not invalid_entries and not duplicate_splits and not invalid_files and not phase_errors
    print(f"\nResultado: {'VALIDO' if is_valid else 'INVALIDO'}")
    return 0 if is_valid else 1


def main():
    parser = argparse.ArgumentParser(
        description="Verifica pacientes e patologias no M&Ms resampled."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    args = parser.parse_args()
    return verify_dataset(args.dataset_dir, args.csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
