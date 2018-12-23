import numpy as np
import pandas as pd
import scipy.stats as sp

# file path
DATA_DIR = "./data"
ORI_DATA_PATH = DATA_DIR + "/diabetic_data.csv"
MAP_PATH = DATA_DIR + "/IDs_mapping.csv"
OUTPUT_DATA_PATH = DATA_DIR + "/preprocessed_data.csv"

# load data
dataframe_ori = pd.read_csv(ORI_DATA_PATH)
NUM_RECORDS = dataframe_ori.shape[0]
NUM_FEATURE = dataframe_ori.shape[1]

# make a copy of the dataframe for preprocessing
df = dataframe_ori.copy(deep=True)

# Drop features

df = df.drop(['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton'], axis=1)
# drop bad data with 3 '?' in diag
drop_ID = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)
# drop died patient data which 'discharge_disposition_id' == 11 | 19 | 20 | 21 indicates 'Expired'
drop_ID = drop_ID.union(set(df[(df['discharge_disposition_id'] == 11) | (df['discharge_disposition_id'] == 19) | \
                               (df['discharge_disposition_id'] == 20) | (df['discharge_disposition_id'] == 21)].index))
# drop 3 data with 'Unknown/Invalid' gender
drop_ID = drop_ID.union(df['gender'][df['gender'] == 'Unknown/Invalid'].index)
new_ID = list(set(df.index) - set(drop_ID))
df = df.iloc[new_ID]

# process readmitted data
df['readmitted'] = df['readmitted'].replace('>30', 2)
df['readmitted'] = df['readmitted'].replace('<30', 1)
df['readmitted'] = df['readmitted'].replace('NO', 0)

# calculate change times through 23 kinds of medicines
# high change times refer to higher prob to readmit
# 'num_med_changed' to counts medicine change
print('\n--Medicine  related--')
medicine = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
            'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin',
            'troglitazone', 'tolbutamide', 'acetohexamide']

for med in medicine:
    tmp = med + 'temp'
    df[tmp] = df[med].apply(lambda x: 1 if (x == 'Down' or x == 'Up') else 0)

# two new feature
df['num_med_changed'] = 0
for med in medicine:
    tmp = med + 'temp'
    df['num_med_changed'] += df[tmp]
    del df[tmp]

for i in medicine:
    df[i] = df[i].replace('Steady', 1)
    df[i] = df[i].replace('No', 0)
    df[i] = df[i].replace('Up', 1)
    df[i] = df[i].replace('Down', 1)
df['num_med_taken'] = 0
for med in medicine:
    print(med)
    df['num_med_taken'] = df['num_med_taken'] + df[med]

# encode race
df['race'] = df['race'].replace('Asian', 0)
df['race'] = df['race'].replace('AfricanAmerican', 1)
df['race'] = df['race'].replace('Caucasian', 2)
df['race'] = df['race'].replace('Hispanic', 3)
df['race'] = df['race'].replace('Other', 4)
df['race'] = df['race'].replace('?', 4)

# map
df['A1Cresult'] = df['A1Cresult'].replace('None', -99)  # -1 -> -99
df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)

df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)  # -1 -> -99

df['change'] = df['change'].replace('No', 0)
df['change'] = df['change'].replace("Ch", 1)

df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)

df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
df['diabetesMed'] = df['diabetesMed'].replace('No', 0)

print('diabetesMed end')

age_dict = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
            '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
df['age'] = df.age.map(age_dict)
df['age'] = df['age'].astype('int64')

print('age end')

# simplify
# admission_type_id : [2, 7] -> 1, [6, 8] -> 5
a, b = [2, 7], [6, 8]
for i in a:
    df['admission_type_id'] = df['admission_type_id'].replace(i, 1)
for j in b:
    df['admission_type_id'] = df['admission_type_id'].replace(j, 5)

# discharge_disposition_id : [6, 8, 9, 13] -> 1, [3, 4, 5, 14, 22, 23, 24] -> 2,
#                            [12, 15, 16, 17] -> 10, [19, 20, 21] -> 11, [25, 26] -> 18
a, b, c, d, e = [6, 8, 9, 13], [3, 4, 5, 14, 22, 23, 24], [12, 15, 16, 17], \
                [19, 20, 21], [25, 26]
for i in a:
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(i, 1)
for j in b:
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(j, 2)
for k in c:
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(k, 10)
# data of died patients have been dropped
# for p in d:
#     df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(p, 11)
for q in e:
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(q, 18)

# admission_source_id : [3, 2] -> 1, [5, 6, 10, 22, 25] -> 4,
#                       [15, 17, 20, 21] -> 9, [13, 14] -> 11
a, b, c, d = [3, 2], [5, 6, 10, 22, 25], [15, 17, 20, 21], [13, 14]
for i in a:
    df['admission_source_id'] = df['admission_source_id'].replace(i, 1)
for j in b:
    df['admission_source_id'] = df['admission_source_id'].replace(j, 4)
for k in c:
    df['admission_source_id'] = df['admission_source_id'].replace(k, 9)
for p in d:
    df['admission_source_id'] = df['admission_source_id'].replace(p, 11)

print('id end')

#  Classify Diagnoses by ICD-9
df.loc[df['diag_1'].str.contains('V', na=False), ['diag_1']] = 0
df.loc[df['diag_1'].str.contains('E', na=False), ['diag_1']] = 0

df['diag_1'] = df['diag_1'].replace('?', -1)

df['diag_1'] = pd.to_numeric(df['diag_1'], errors='coerce')

for index, row in df.iterrows():
    if (row['diag_1'] >= 1 and row['diag_1'] <= 139):
        df.loc[index, 'diag_1'] = 1
    elif (row['diag_1'] >= 140 and row['diag_1'] <= 239):
        df.loc[index, 'diag_1'] = 2
    elif (row['diag_1'] >= 240 and row['diag_1'] <= 279):
        df.loc[index, 'diag_1'] = 3
    elif (row['diag_1'] >= 280 and row['diag_1'] <= 289):
        df.loc[index, 'diag_1'] = 4
    elif (row['diag_1'] >= 290 and row['diag_1'] <= 319):
        df.loc[index, 'diag_1'] = 5
    elif (row['diag_1'] >= 320 and row['diag_1'] <= 389):
        df.loc[index, 'diag_1'] = 6
    elif (row['diag_1'] >= 390 and row['diag_1'] <= 459):
        df.loc[index, 'diag_1'] = 7
    elif (row['diag_1'] >= 460 and row['diag_1'] <= 519):
        df.loc[index, 'diag_1'] = 8
    elif (row['diag_1'] >= 520 and row['diag_1'] <= 579):
        df.loc[index, 'diag_1'] = 9
    elif (row['diag_1'] >= 580 and row['diag_1'] <= 629):
        df.loc[index, 'diag_1'] = 10
    elif (row['diag_1'] >= 630 and row['diag_1'] <= 679):
        df.loc[index, 'diag_1'] = 11
    elif (row['diag_1'] >= 680 and row['diag_1'] <= 709):
        df.loc[index, 'diag_1'] = 12
    elif (row['diag_1'] >= 710 and row['diag_1'] <= 739):
        df.loc[index, 'diag_1'] = 13
    elif (row['diag_1'] >= 740 and row['diag_1'] <= 759):
        df.loc[index, 'diag_1'] = 14
    elif (row['diag_1'] >= 760 and row['diag_1'] <= 779):
        df.loc[index, 'diag_1'] = 15
    elif (row['diag_1'] >= 780 and row['diag_1'] <= 799):
        df.loc[index, 'diag_1'] = 16
    elif (row['diag_1'] >= 800 and row['diag_1'] <= 999):
        df.loc[index, 'diag_1'] = 17

print('diag_1 end')
df.loc[df['diag_2'].str.contains('V', na=False), ['diag_2']] = 0
df.loc[df['diag_2'].str.contains('E', na=False), ['diag_2']] = 0

df['diag_2'] = df['diag_2'].replace('?', -1)

df['diag_2'] = pd.to_numeric(df['diag_2'], errors='coerce')

for index, row in df.iterrows():
    if (row['diag_2'] >= 1 and row['diag_2'] <= 139):
        df.loc[index, 'diag_2'] = 1
    elif (row['diag_2'] >= 140 and row['diag_2'] <= 239):
        df.loc[index, 'diag_2'] = 2
    elif (row['diag_2'] >= 240 and row['diag_2'] <= 279):
        df.loc[index, 'diag_2'] = 3
    elif (row['diag_2'] >= 280 and row['diag_2'] <= 289):
        df.loc[index, 'diag_2'] = 4
    elif (row['diag_2'] >= 290 and row['diag_2'] <= 319):
        df.loc[index, 'diag_2'] = 5
    elif (row['diag_2'] >= 320 and row['diag_2'] <= 389):
        df.loc[index, 'diag_2'] = 6
    elif (row['diag_2'] >= 390 and row['diag_2'] <= 459):
        df.loc[index, 'diag_2'] = 7
    elif (row['diag_2'] >= 460 and row['diag_2'] <= 519):
        df.loc[index, 'diag_2'] = 8
    elif (row['diag_2'] >= 520 and row['diag_2'] <= 579):
        df.loc[index, 'diag_2'] = 9
    elif (row['diag_2'] >= 580 and row['diag_2'] <= 629):
        df.loc[index, 'diag_2'] = 10
    elif (row['diag_2'] >= 630 and row['diag_2'] <= 679):
        df.loc[index, 'diag_2'] = 11
    elif (row['diag_2'] >= 680 and row['diag_2'] <= 709):
        df.loc[index, 'diag_2'] = 12
    elif (row['diag_2'] >= 710 and row['diag_2'] <= 739):
        df.loc[index, 'diag_2'] = 13
    elif (row['diag_2'] >= 740 and row['diag_2'] <= 759):
        df.loc[index, 'diag_2'] = 14
    elif (row['diag_2'] >= 760 and row['diag_2'] <= 779):
        df.loc[index, 'diag_2'] = 15
    elif (row['diag_2'] >= 780 and row['diag_2'] <= 799):
        df.loc[index, 'diag_2'] = 16
    elif (row['diag_2'] >= 800 and row['diag_2'] <= 999):
        df.loc[index, 'diag_2'] = 17
print('diag_2 end')

df.loc[df['diag_3'].str.contains('V', na=False), ['diag_3']] = 0
df.loc[df['diag_3'].str.contains('E', na=False), ['diag_3']] = 0

df['diag_3'] = df['diag_3'].replace('?', -1)

df['diag_3'] = pd.to_numeric(df['diag_3'], errors='coerce')

for index, row in df.iterrows():
    if (row['diag_3'] >= 1 and row['diag_3'] <= 139):
        df.loc[index, 'diag_3'] = 1
    elif (row['diag_3'] >= 140 and row['diag_3'] <= 239):
        df.loc[index, 'diag_3'] = 2
    elif (row['diag_3'] >= 240 and row['diag_3'] <= 279):
        df.loc[index, 'diag_3'] = 3
    elif (row['diag_3'] >= 280 and row['diag_3'] <= 289):
        df.loc[index, 'diag_3'] = 4
    elif (row['diag_3'] >= 290 and row['diag_3'] <= 319):
        df.loc[index, 'diag_3'] = 5
    elif (row['diag_3'] >= 320 and row['diag_3'] <= 389):
        df.loc[index, 'diag_3'] = 6
    elif (row['diag_3'] >= 390 and row['diag_3'] <= 459):
        df.loc[index, 'diag_3'] = 7
    elif (row['diag_3'] >= 460 and row['diag_3'] <= 519):
        df.loc[index, 'diag_3'] = 8
    elif (row['diag_3'] >= 520 and row['diag_3'] <= 579):
        df.loc[index, 'diag_3'] = 9
    elif (row['diag_3'] >= 580 and row['diag_3'] <= 629):
        df.loc[index, 'diag_3'] = 10
    elif (row['diag_3'] >= 630 and row['diag_3'] <= 679):
        df.loc[index, 'diag_3'] = 11
    elif (row['diag_3'] >= 680 and row['diag_3'] <= 709):
        df.loc[index, 'diag_3'] = 12
    elif (row['diag_3'] >= 710 and row['diag_3'] <= 739):
        df.loc[index, 'diag_3'] = 13
    elif (row['diag_3'] >= 740 and row['diag_3'] <= 759):
        df.loc[index, 'diag_3'] = 14
    elif (row['diag_3'] >= 760 and row['diag_3'] <= 779):
        df.loc[index, 'diag_3'] = 15
    elif (row['diag_3'] >= 780 and row['diag_3'] <= 799):
        df.loc[index, 'diag_3'] = 16
    elif (row['diag_3'] >= 800 and row['diag_3'] <= 999):
        df.loc[index, 'diag_3'] = 17
print('diag_3 end')

# df['new_1'] = df['num_medications'] * df['time_in_hospital']
# # df['add_feature_2'] = df['change'] * df['num_medications']
# df['new_3'] = df['age'] * df['number_diagnoses']

print('diag end')

def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0))


numerics = ['race', 'age', 'time_in_hospital', 'num_medications', 'number_diagnoses',
            'num_med_changed', 'num_med_taken', 'number_inpatient', 'number_outpatient', 'number_emergency',
            'num_procedures', 'num_lab_procedures']

df[numerics] = standardize(df[numerics])
df = df[(np.abs(sp.stats.zscore(df[numerics])) < 3).all(axis=1)]

print('begin out')
print(OUTPUT_DATA_PATH)
df.to_csv(OUTPUT_DATA_PATH)
