import sys
import pandas as pd
import re
import os
from sklearn import preprocessing, f1_score
import pickle

PATH = r"/home/student/Early-Prediction-of-Sepsis-"


def load_files(path_folder):
    df_patients_train = []
    for i, filename in enumerate(os.listdir(path_folder)):
        f = os.path.join(path_folder, filename)
        df_patient = pd.read_csv(str(f), sep='|')
        try:
            first_one = df_patient.loc[df_patient["SepsisLabel"] == 1].index[
                            0] + 1  # extracts the first index when the label changes from 0 to 1
        except IndexError:
            first_one = len(df_patient)  # in case the patient is not sick, return the full df of the patient
        df_patient_filtered = df_patient.iloc[0:first_one]
        patient_id_number = str(re.search(r'\d+', filename).group())
        patient_id = 'patient_' + patient_id_number
        df_patient_filtered["patient id"] = patient_id
        df_patients_train.append(df_patient_filtered)
    df = pd.concat(df_patients_train)
    return df


def main(argv):
    path_folder = argv[1]  # gets the path of the folder. Folder train or test
    test_df = load_files(path_folder)
    test_df_aggregated = test_df.groupby("patient id").mean()
    test_df_aggregated.loc[test_df_aggregated["SepsisLabel"] > 0, "SepsisLabel"] = 1
    test_df_aggregated_filtered = test_df_aggregated[
        ['ICULOS',
         'HR',
         'MAP',
         'Resp',
         'EtCO2',
         'BaseExcess',
         'FiO2',
         'pH',
         'PaCO2',
         'Lactate',
         'SepsisLabel']
    ]

    # for the features with the most null values (the treatments) we will new binary features
    convert_to_binary = ['EtCO2', 'BaseExcess', 'FiO2', 'pH', 'PaCO2', 'Lactate']
    for feature in convert_to_binary:
        test_df_aggregated_filtered['Bin_' + feature] = test_df_aggregated_filtered[feature].apply(
            lambda x: 0 if pd.isna(x) else 1)

    convert_to_mean = ["HR", "MAP", "Resp"]
    for feature in convert_to_mean:
        test_df_aggregated_filtered[feature] = test_df_aggregated_filtered[feature].fillna(
            test_df_aggregated_filtered[feature].mean())

    test_df_aggregated_filtered_not_nan = test_df_aggregated_filtered.drop(convert_to_binary, axis=1)
    y_true = test_df_aggregated_filtered_not_nan[['SepsisLabel']]
    X_without_label_test = test_df_aggregated_filtered_not_nan.drop('SepsisLabel', axis=1)  # Predictors
    X_test_file = preprocessing.scale(X_without_label_test)

    # load model
    filepath = PATH+r"/xgb_classifier_2_model.sav"
    xgb_best = pickle.load(open(filepath, 'rb'))

    predictions_final = xgb_best.predict(X_test_file)
    print("f1_score : ",f1_score(y_true,predictions_final))
    # Save results to a csv
    results = pd.DataFrame(list(zip(list(X_without_label_test.reset_index()['patient id']), list(predictions_final))),
                           columns=['id', 'prediction'])

    # Sort dataframe by 'patient id' column using the suffixes
    results['suffix'] = results['id'].apply(lambda row: int(row.split('_')[1]))
    results = results.sort_values('suffix')
    results = results.drop('suffix', axis=1)
    results.to_csv(PATH+r"/prediction.csv", index=False)
    print("Done. The prediciton csv file is in your directory")


if __name__ == "__main__":
    main(sys.argv)
