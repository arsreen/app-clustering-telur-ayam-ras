import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df, fitur):
    """
    Membersihkan data dari nilai kosong (NaN) pada kolom fitur yang digunakan.
    Menghapus baris yang memiliki missing value agar tidak error di model clustering.
    """
    df_clean = df.dropna(subset=fitur)
    return df_clean

def scale_features(df, fitur):
    """
    Normalisasi fitur menggunakan MinMaxScaler agar seluruh variabel dalam rentang 0-1.
    """
    X = df[fitur].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
