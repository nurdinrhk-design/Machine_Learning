import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

# Model Selection & Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# Mengabaikan warning yang tidak krusial
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print("--- Memulai Pertemuan 6: Analisis Mendalam RF ---")
print("="*50)

# Langkah 1: Muat Data & Split
try:
    df = pd.read_csv("processed_kelulusan.csv")
    print("Berhasil memuat 'processed_kelulusan.csv'")
except FileNotFoundError:
    print("Error: File 'processed_kelulusan.csv' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'pertemuan_4.py' terlebih dahulu.")
    exit()

X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nDataset berhasil di-split:")
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# Langkah 2: Baseline Model & Pipeline
num_cols = X_train.select_dtypes(include=np.number).columns
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", preprocessor), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_pred = pipe_rf.predict(X_val)
print("\nBaseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# Langkah 3: Validasi Silang
print("\n--- Validasi Silang (P6) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_rf, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (train baseline RF): {scores.mean():.4f} ± {scores.std():.4f}")

# Langkah 4: Tuning Ringkas (GridSearch)
print("\n--- Tuning: GridSearch (P6) ---")
param = {
  "clf__max_depth": [None, 10, 20],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

print(f"Best params (GridSearch): {gs.best_params_}")
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print(f"Best RF — F1(val) (setelah tuning): {f1_score(y_val, y_val_best, average='macro'):.4f}")

# Langkah 5: Evaluasi Akhir (Test Set)
print("\n--- Evaluasi Akhir: Test Set (P6) ---")
final_model = best_model
y_test_pred = final_model.predict(X_test)

print("F1 Macro (test):", f1_score(y_test, y_test_pred, average="macro"))
print("Classification Report (test):")
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Plot ROC dan PR (P6, L5)
try:
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    print("ROC-AUC (test):", roc_auc_score(y_test, y_test_proba))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_test_proba):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set) - P6"); plt.legend()
    plt.tight_layout(); plt.savefig("p6_roc_test.png", dpi=120)
    plt.close()
    print("Plot 'p6_roc_test.png' disimpan.")

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec, label="PR Curve")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set) - P6"); plt.legend()
    plt.tight_layout(); plt.savefig("p6_pr_test.png", dpi=120)
    plt.close()
    print("Plot 'p6_pr_test.png' disimpan.")
except Exception as e:
    print(f"Gagal membuat plot ROC/PR: {e}")

# Langkah 6: Pentingnya Fitur
print("\n--- Feature Importance (P6) ---")
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    fn_cleaned = [name.split('__')[-1] for name in fn]
    
    top = sorted(zip(fn_cleaned, importances), key=lambda x: x[1], reverse=True)
    print("Top feature importance (Gini):")
    for name, val in top:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print(f"Feature importance tidak tersedia: {e}")

# Langkah 7: Simpan Model
joblib.dump(final_model, "rf_model_p6.pkl")
print("\nModel disimpan sebagai 'rf_model_p6.pkl'")

# Langkah 8: Cek Inference Lokal
mdl = joblib.load("rf_model_p6.pkl")
sample_data = {
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}
all_cols_sample = {col: 0 for col in X_train.columns}
all_cols_sample.update(sample_data)
sample_df = pd.DataFrame([all_cols_sample])
prediksi = int(mdl.predict(sample_df)[0])
print(f"Contoh Prediksi (P6) untuk IPK 3.4: {prediksi} (1=Lulus, 0=Gagal)")


print("\n" + "="*50)
print("--- PERTEMUAN 6 SELESAI ---")
print("="*50)