import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing(input_path, output_path):
    print(f"1. Membaca data dari {input_path}...")
    column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    df = pd.read_csv(input_path, header=None, names=column_names)
    
    print("2. Melakukan pembersihan data...")
    df = df.drop_duplicates()
    df = df.drop('id', axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    print("3. Standarisasi fitur...")
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"4. Menyimpan data ke {output_path}...")
    df_final = X_scaled.copy()
    df_final['diagnosis'] = y.values
    
    # Memastikan folder tujuan ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print("Otomatisasi Preprocessing Selesai!")

if __name__ == "__main__":
    # Path disesuaikan agar bisa dijalankan dari root GitHub (Langkah 4)
    RAW_DATA_PATH = "wdbc_raw/wdbc.data"
    PROCESSED_DATA_PATH = "preprocessing/wdbc_preprocessing/wdbc_preprocessed.csv"
    run_preprocessing(RAW_DATA_PATH, PROCESSED_DATA_PATH)