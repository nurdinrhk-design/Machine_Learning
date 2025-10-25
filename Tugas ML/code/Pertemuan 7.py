import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Model Selection & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Mengabaikan warning yang tidak krusial
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print("--- Memulai Pertemuan 7: Artificial Neural Network ---")
print("="*50)

# Langkah 1: Siapkan Data
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print("Berhasil memuat 'processed_kelulusan.csv'")
except FileNotFoundError:
    print("Error: File 'processed_kelulusan.csv' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'pertemuan_4.py' terlebih dahulu.")
    exit()

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nDataset berhasil di-split.")

# PERBAIKAN: Scaling SETELAH split
print("Menyiapkan data untuk ANN (Scaling data...)")
sc_ann = StandardScaler()
X_train_s = sc_ann.fit_transform(X_train) # Fit HANYA di train
X_val_s = sc_ann.transform(X_val)
X_test_s = sc_ann.transform(X_test)

# Konversi y ke numpy array (Keras lebih suka numpy)
y_train_s = y_train.values
y_val_s = y_val.values
y_test_s = y_test.values

# Langkah 2: Bangun Model ANN
tf.random.set_seed(42)
model_ann = keras.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # klasifikasi biner
])

model_ann.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])
print("\nModel ANN Summary (Langkah 2):")
model_ann.summary()

# Langkah 3: Training dengan Early Stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_auc", # Monitor AUC di val set
    patience=20,       # Naikkan patience
    restore_best_weights=True,
    mode='max' # Karena AUC, kita ingin 'max'
)

print("\nMemulai training ANN (Langkah 3)...")
history = model_ann.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=200, # Naikkan epochs, biarkan EarlyStopping bekerja
    batch_size=8, # Kecilkan batch size untuk data kecil
    callbacks=[es],
    verbose=0 # Set ke 0 agar tidak terlalu ramai
)
print("Training ANN Selesai.")

# Langkah 4: Evaluasi di Test Set
print("\n--- Evaluasi ANN di Test Set (Langkah 4) ---")
loss, acc, auc = model_ann.evaluate(X_test_s, y_test_s, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

y_proba = model_ann.predict(X_test_s).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix (ANN Test):")
print(confusion_matrix(y_test_s, y_pred))
print("\nClassification Report (ANN Test):")
print(classification_report(y_test_s, y_pred, digits=3))

# Langkah 5: Visualisasi Learning Curve
print("Membuat plot 'p7_learning_curve.png' (Langkah 5)...")
plt.figure(figsize=(10, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve (Loss)")

# Plot AUC
plt.subplot(1, 2, 2)
plt.plot(history.history["auc"], label="Train AUC")
plt.plot(history.history["val_auc"], label="Val AUC")
plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.legend()
plt.title("Learning Curve (AUC)")

plt.tight_layout()
plt.savefig("p7_learning_curve.png", dpi=120)
plt.close()
print("Plot 'p7_learning_curve.png' disimpan.")

print("\n" + "="*50)
print("--- PERTEMUAN 7 SELESAI ---")
print("="*50)