from Classification3D.preprocessing.load_mms import load_mmms_data

volumes, labels, patient_data = load_mmms_data()
print(volumes.shape, labels, patient_data)
