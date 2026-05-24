import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Fungsi untuk memuat dataset."""
    print(f"Membaca data dari: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_and_preprocess(df):
    """Fungsi untuk membersihkan dan memproses data."""
    print("Memulai proses preprocessing...")
    
    # 1. Menghapus missing values jika ada
    df_clean = df.dropna().copy()
    
    # 2. Memisahkan fitur (X) dan target (y)
    # Asumsi kolom target pada Wine Quality adalah 'quality'
    X = df_clean.drop('quality', axis=1)
    y = df_clean['quality']
    
    # 3. Standarisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 4. Menggabungkan kembali fitur yang sudah diskalakan dengan target
    df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    
    print("Preprocessing selesai.")
    return df_final

def save_data(df, output_path):
    """Fungsi untuk menyimpan data yang sudah bersih."""
    print(f"Menyimpan data bersih ke: {output_path}")
    df.to_csv(output_path, index=False)
    print("Data berhasil disimpan!")

if __name__ == "__main__":
    # --- Konfigurasi Path ---
    # Path disesuaikan dengan asumsi script dijalankan dari root folder
    INPUT_FILE = "dataset_raw/wine_quality.csv"
    OUTPUT_FOLDER = "dataset_preprocessing"
    OUTPUT_FILE = f"{OUTPUT_FOLDER}/wine_quality_clean.csv"
    
    # Membuat folder output jika belum ada
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # --- Eksekusi Pipeline Preprocessing ---
    try:
        # Step 1: Load Data
        raw_data = load_data(INPUT_FILE)
        
        # Step 2: Preprocess Data
        processed_data = clean_and_preprocess(raw_data)
        
        # Step 3: Save Data
        save_data(processed_data, OUTPUT_FILE)
        
    except Exception as e:
        print(f"Terjadi kesalahan pada pipeline: {e}")