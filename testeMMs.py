from Classification3D.preprocessing.load_mms import load_mms_data

volumes, labels, patient_data = load_mms_data()
# print(volumes.shape, labels, patient_data)
print(labels)
