import pandas as pd


biological_data = pd.DataFrame({
    'Patient': ['pd002', 'pd003', 'pd004', 'pd005', 'pd006', 'pd007', 'pd008', 'el001', 'el002', 'el003', 'el004', 'el006', 'el007', 'el008', 'el009', 'el010', 's001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013'],
    'Height': [172, 174, 170, 160, 182, 174, 175, 168, 173, 167, 165, 172, 170, 177, 173, 185, 169, 169, 173, 178, 175, 176, 180, 182, 182, 185, 170, 168, 187],
    'Age': [76, 73, 79, 65, 63, 67, 71, 85, 96, 67, 60, 61, 83, 67, 78, 65, 20, 34, 33, 32, 50, 30, 43, 27, 51, 56, 30, 30, 58]
})


def get_height(patient_id):
    height = biological_data.loc[biological_data['Patient'] == patient_id, 'Height'].values[0]
    return height

def get_age(patient_id):
    age = biological_data.loc[biological_data['Patient'] == patient_id, 'Age'].values[0]
    return age