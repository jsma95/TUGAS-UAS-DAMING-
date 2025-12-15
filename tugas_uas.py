import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("AirQuality.csv", sep=';')

print("\n===== INFO DATA =====")
print(df.info())

print("\n===== DATA AWAL =====")
print(df.head())

print("\n===== DESKRIPSI DATA =====")
print(df.describe())

# ===============================
# CLEANING DATA
# ===============================
df = df.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], errors='ignore')

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype(float)

df = df.replace(-200, np.nan)
df = df.fillna(df.median())

print("\n===== DATA SETELAH CLEANING =====")
print(df.describe())

# ===============================
# SCALING
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("\n===== DATA SETELAH SCALING =====")
print(df_scaled.head())

# ===============================
# K-MEANS CLUSTERING
# ===============================
from sklearn.cluster import KMeans

wcss = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    wcss.append(km.inertia_)
    print(f"WCSS untuk k={k}: {km.inertia_}")

plt.plot(range(2, 8), wcss, marker='o')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

print("\n===== DISTRIBUSI CLUSTER =====")
print(df['Cluster'].value_counts())

# ===============================
# EVALUASI CLUSTER
# ===============================
from sklearn.metrics import silhouette_score

score = silhouette_score(df_scaled, df['Cluster'])
print("\nSilhouette Score:", score)

# ===============================
# PCA VISUALISASI
# ===============================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

print("\n===== HASIL PCA =====")
print(df[['PCA1', 'PCA2', 'Cluster']].head())

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Visualisasi Cluster (PCA)')
plt.show()

# ===============================
# LOGISTIC REGRESSION
# ===============================
X = df_scaled
y = df['Cluster']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

logreg.fit(X_train, y_train)

# ===============================
# EVALUASI MODEL
# ===============================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = logreg.predict(X_test)

print("\n===== HASIL LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===============================
# KOEFISIEN MODEL
# ===============================
coef_df = pd.DataFrame(
    logreg.coef_,
    columns=X.columns,
    index=[f'Cluster {i}' for i in logreg.classes_]
)

print("\n===== KOEFISIEN LOGISTIC REGRESSION =====")
print(coef_df)

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("model", exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(kmeans, "model/kmeans.pkl")
joblib.dump(logreg, "model/logreg.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

# ===============================
# SAVE OUTPUT (OPSIONAL, TAPI DISARANKAN)
# ===============================
coef_df.to_csv("model/koefisien_logreg.csv")
df.to_csv("model/data_dengan_cluster.csv", index=False)

print("\n✅ SEMUA PROSES SELESAI & FILE DISIMPAN")
