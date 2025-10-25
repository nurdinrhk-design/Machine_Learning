import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

# Model Selection & Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Mengabaikan warning yang tidak krusial
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print("--- Memulai Pertemuan 5: Modeling & Tuning ---")
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

# 70% Train, 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 30% Temp -> 15% Val, 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nDataset berhasil di-split:")
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# Langkah 2: Baseline Model & Pipeline
num_cols = X_train.select_dtypes(include=np.number).columns

# Definisikan Preprocessor
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

# 1. Baseline Logistic Regression (P5, L2)
print("\n--- Model 1: Baseline Logistic Regression (P5) ---")
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", preprocessor), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val)
print("Baseline (LogReg) F1 Macro (val):", f1_score(y_val, y_val_pred_lr, average="macro"))
print("Classification Report (LogReg Val):")
print(classification_report(y_val, y_val_pred_lr, digits=3))

# 2. Model Alternatif Random Forest (P5, L3)
print("\n--- Model 2: Baseline Random Forest (P5) ---")
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", preprocessor), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("Baseline (RF) F1 Macro (val):", f1_score(y_val, y_val_rf, average="macro"))

# 3. Tuning RF (P5, L4)
print("\n--- Tuning: GridSearch Random Forest (P5) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param = {
  "clf__max_depth": [None, 10, 20],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

print(f"Best params (GridSearch): {gs.best_params_}")
print(f"Best CV F1 (GridSearch): {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print(f"Best RF F1(val) (setelah tuning): {f1_score(y_val, y_val_best, average='macro'):.4f}")

# 4. Evaluasi Akhir di Test Set (P5, L5)
print("\n--- Evaluasi Akhir: Test Set (RF Model) ---")
final_model = best_rf 
y_test_pred = final_model.predict(X_test)

print("F1 Macro (test):", f1_score(y_test, y_test_pred, average="macro"))
print("Classification Report (test):")
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Plot ROC
try:
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    print("ROC-AUC (test):", roc_auc_score(y_test, y_test_proba))

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_test_proba):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set) - P5")
    plt.legend()
    plt.tight_layout()
    plt.savefig("p5_roc_test.png", dpi=120)
    plt.close()
    print("Plot 'p5_roc_test.png' disimpan.")
except Exception as e:
    print(f"Gagal membuat plot ROC: {e}")

# 5. Simpan Model (P5, L6)
joblib.dump(final_model, "model_p5.pkl")
print("\nModel disimpan sebagai 'model_p5.pkl'")

print("\n" + "="*50)
print("--- PERTEMUAN 5 SELESAI ---")
print("="*50)