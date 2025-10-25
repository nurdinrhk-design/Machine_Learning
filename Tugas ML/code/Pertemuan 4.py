import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from io import StringIO

# Mengabaikan warning yang tidak krusial
warnings.filterwarnings('ignore')

def create_dataset():
    """
    Membuat dataset yang lebih besar untuk menghindari error split/CV.
    Menggunakan 10 data asli + 40 data sintetis.
    """
    # 10 data asli dari soal
    original_data = """IPK,Jumlah_Absensi,Waktu_Belajar_Jam,Lulus
3.8,3,10,1
2.5,8,5,0
3.4,4,7,1
2.1,12,2,0
3.9,2,12,1
2.8,6,4,0
3.2,5,8,1
2.7,7,3,0
3.6,4,9,1
2.3,9,4,0
"""
    df_original = pd.read_csv(StringIO(original_data))

    # 40 data sintetis untuk menambah ukuran
    np.random.seed(42)
    # 20 Lulus (IPK tinggi, Absen rendah, Waktu belajar tinggi)
    ipk_lulus = np.random.uniform(3.0, 4.0, 20)
    absen_lulus = np.random.randint(0, 6, 20)
    waktu_lulus = np.random.randint(6, 16, 20)
    
    # 20 Gagal (IPK rendah, Absen tinggi, Waktu belajar rendah)
    ipk_gagal = np.random.uniform(2.0, 3.2, 20)
    absen_gagal = np.random.randint(5, 16, 20)
    waktu_gagal = np.random.randint(0, 7, 20)

    data_sintetis = {
        'IPK': np.concatenate([ipk_lulus, ipk_gagal]),
        'Jumlah_Absensi': np.concatenate([absen_lulus, absen_gagal]),
        'Waktu_Belajar_Jam': np.concatenate([waktu_lulus, waktu_gagal]),
        'Lulus': np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
    }
    df_sintetis = pd.DataFrame(data_sintetis)
    
    # Gabungkan dan acak
    df = pd.concat([df_original, df_sintetis]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset berhasil dibuat dengan total {len(df)} baris.")
    return df

print("\n" + "="*50)
print("--- Memulai Pertemuan 4: Persiapan Data ---")
print("="*50)

# Langkah 1: Buat Dataset
df = create_dataset()
df.to_csv("kelulusan_mahasiswa.csv", index=False)
print("File 'kelulusan_mahasiswa.csv' berhasil dibuat (50 baris).")

# Langkah 2: Collection
print("\nInfo Dataset (Langkah 2):")
print(df.info())
print("\nHead Dataset (Langkah 2):")
print(df.head())

# Langkah 3: Cleaning
print("\nMissing Values (Langkah 3):")
print(df.isnull().sum())

df = df.drop_duplicates()
print(f"Data setelah drop_duplicates: {df.shape}")

print("Membuat plot 'p4_boxplot_ipk.png' (Langkah 3)...")
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.savefig("p4_boxplot_ipk.png", dpi=120)
plt.close()

# Langkah 4: Exploratory Data Analysis (EDA)
print("\nStatistik Deskriptif (Langkah 4):")
print(df.describe())

print("Membuat plot 'p4_hist_ipk.png' (Langkah 4)...")
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.savefig("p4_hist_ipk.png", dpi=120)
plt.close()

print("Membuat plot 'p4_scatter.png' (Langkah 4)...")
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar")
plt.savefig("p4_scatter.png", dpi=120)
plt.close()

print("Membuat plot 'p4_heatmap.png' (Langkah 4)...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi")
plt.savefig("p4_heatmap.png", dpi=120)
plt.close()

# Langkah 5: Feature Engineering
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14 # Asumsi 14 pertemuan
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)
print("\nFeature Engineering selesai. Data disimpan ke 'processed_kelulusan.csv'")
print(df.head())

print("\n" + "="*50)
print("--- PERTEMUAN 4 SELESAI ---")
print("File 'processed_kelulusan.csv' siap untuk Pertemuan 5, 6, dan 7.")
print("="*50)