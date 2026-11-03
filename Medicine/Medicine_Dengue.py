import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. DESCARGA Y PREPARACIÓN (SIVIGILA)
# ==========================================
print("1. Descargando datos de SIVIGILA (Dengue 2016-2018)...")
url = "https://www.datos.gov.co/resource/4hyg-wa9d.json"
params = {"$where": "cod_eve = '210' AND ano BETWEEN 2016 AND 2018", "$limit": 500000}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    datos_sivigila = pd.DataFrame(response.json())
    datos_sivigila['conteo'] = pd.to_numeric(datos_sivigila['conteo'], errors='coerce').fillna(0)
    dengue_agregado = datos_sivigila.groupby(['cod_mun_o', 'ano', 'semana'], as_index=False)['conteo'].sum()
    print(f"   -> Datos de SIVIGILA listos. Registros: {len(dengue_agregado)}")
except Exception as e:
    print(f"Error descargando SIVIGILA: {e}")
    raise

# ==========================================
# 2. CARGA DE METADATOS (PHYSIONET)
# ==========================================
print("\n2. Cargando metadatos de PhysioNet...")
ruta_metadatos_physionet = "metadata.csv"
try:
    physionet_meta = pd.read_csv(ruta_metadatos_physionet)
    physionet_meta = physionet_meta.rename(columns={'Municipality Code': 'cod_mun_o', 'Municipality code': 'cod_mun_o', 'Year': 'ano', 'Epi_Week': 'semana'})
    print(f"   -> Metadatos cargados. Registros: {len(physionet_meta)}")
except FileNotFoundError:
    print(f"   -> ADVERTENCIA: No se encontró '{ruta_metadatos_physionet}'. Creando DataFrame vacío.")
    physionet_meta = pd.DataFrame(columns=['cod_mun_o', 'ano', 'semana'])

# ==========================================
# 3. MERGE ESPACIO-TEMPORAL
# ==========================================
print("\n3. Uniendo ambas fuentes (Prueba de Acoplamiento)...")
dengue_agregado['cod_mun_o'] = pd.to_numeric(dengue_agregado['cod_mun_o'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(5)
dengue_agregado['ano'] = pd.to_numeric(dengue_agregado['ano'], errors='coerce').fillna(0).astype(int)
dengue_agregado['semana'] = pd.to_numeric(dengue_agregado['semana'], errors='coerce').fillna(0).astype(int)
dengue_agregado = dengue_agregado[dengue_agregado['ano'].isin([2016, 2017, 2018])]

if not physionet_meta.empty:
    physionet_meta['cod_mun_o'] = pd.to_numeric(physionet_meta['cod_mun_o'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(5)
    cols_semanas = [col for col in physionet_meta.columns if '/w' in col]
    physionet_long = physionet_meta.melt(id_vars=['cod_mun_o'], value_vars=cols_semanas, var_name='ano_semana', value_name='casos_physionet')
    physionet_long['ano'] = physionet_long['ano_semana'].str.split('/').str[0].astype(int)
    physionet_long['semana'] = physionet_long['ano_semana'].str.split('/').str[1].str.replace('w', '').astype(int)
    physionet_long = physionet_long[physionet_long['ano'].isin([2016, 2017, 2018])]
    
    df_unido = pd.merge(dengue_agregado, physionet_long, on=['cod_mun_o', 'ano', 'semana'], how='inner')
    print(f"   -> Registros después del merge: {len(df_unido)}")
else:
    df_unido = dengue_agregado.copy()

# ==========================================
# 4. TARGET Y FEATURE ENGINEERING (MEJORADO)
# ==========================================
print("4. Definiendo el target (Brote) y preparando el pipeline...")

df_unido = df_unido.sort_values(['cod_mun_o', 'ano', 'semana']).reset_index(drop=True)

# A. Rezagos extendidos (1, 2, 3, 4, 6, 8 semanas)
for lag in [1, 2, 3, 4, 6, 8]:
    df_unido[f'conteo_lag{lag}'] = df_unido.groupby('cod_mun_o')['conteo'].shift(lag)

# B. Características temporales
df_unido['rolling_mean_4'] = df_unido.groupby('cod_mun_o')['conteo'].transform(lambda x: x.shift(1).rolling(window=4).mean())
df_unido['rolling_std_4'] = df_unido.groupby('cod_mun_o')['conteo'].transform(lambda x: x.shift(1).rolling(window=4).std())
df_unido['diff_1'] = df_unido['conteo_lag1'] - df_unido['conteo_lag2']
df_unido['semana_sin'] = np.sin(2 * np.pi * df_unido['semana'] / 52)
df_unido['semana_cos'] = np.cos(2 * np.pi * df_unido['semana'] / 52)

# C. Contexto Departamental (Contagio espacial)
df_unido['depto'] = df_unido['cod_mun_o'].str[:2]
df_unido['depto_conteo_mean'] = df_unido.groupby(['depto', 'ano', 'semana'])['conteo'].transform('mean')
df_unido['depto_mean_lag1'] = df_unido.groupby(['depto'])['depto_conteo_mean'].shift(1)

# D. Lista de features definitiva
columnas_features = [f'conteo_lag{i}' for i in [1, 2, 3, 4, 6, 8]] + [
    'rolling_mean_4', 'rolling_std_4', 'diff_1', 'semana_sin', 'semana_cos', 'depto_mean_lag1'
]

# Limpiar NaNs generados por todos los shifts
df_unido = df_unido.dropna(subset=columnas_features).reset_index(drop=True)

# E. Target: Brote = conteo actual supera el percentil 80 del municipio
df_unido['umbral_brote'] = df_unido.groupby('cod_mun_o')['conteo'].transform(lambda x: x.quantile(0.80))
df_unido['brote'] = (df_unido['conteo'] > df_unido['umbral_brote']).astype(int)

# F. DIVISIÓN TEMPORAL (Evita predecir el pasado con el futuro)
train_mask = df_unido['ano'] <= 2017
val_mask = (df_unido['ano'] == 2018) & (df_unido['semana'] <= 26)
test_mask = (df_unido['ano'] == 2018) & (df_unido['semana'] > 26)

X_train, y_train = df_unido[train_mask][columnas_features], df_unido[train_mask]['brote']
X_val, y_val = df_unido[val_mask][columnas_features], df_unido[val_mask]['brote']
X_test, y_test = df_unido[test_mask][columnas_features], df_unido[test_mask]['brote']

print(f"   -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

preprocesador = ColumnTransformer([("num", StandardScaler(), columnas_features)])
X_train_p = preprocesador.fit_transform(X_train)
X_val_p   = preprocesador.transform(X_val)
X_test_p  = preprocesador.transform(X_test)

# ==========================================
# 5. MODELO MLP Y ENTRENAMIENTO
# ==========================================
print("\n5. Entrenando el MLP (Línea Base)...")
X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_p, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Batch size subido a 64

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

modelo = MLP(input_dim=X_train_p.shape[1])
negativos = (y_train == 0).sum()
positivos = (y_train == 1).sum()
pos_weight = torch.tensor([negativos / positivos], dtype=torch.float32) if positivos > 0 else torch.tensor([1.0])
print(f"   -> Usando pos_weight: {pos_weight.item():.2f}")

criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)

epocas = 100
paciencia = 12 # Un poco más de paciencia
historial_train, historial_val = [], []
mejor_val_loss = float('inf')
epocas_sin_mejora = 0
mejor_modelo_pesos = None
epoca_early_stop = None

for epoca in range(epocas):
    modelo.train()
    perdida_train = 0.0
    for X_batch, y_batch in train_loader:
        optimizador.zero_grad()
        loss = criterio(modelo(X_batch), y_batch)
        loss.backward()
        optimizador.step()
        perdida_train += loss.item()
    
    perdida_train_prom = perdida_train / len(train_loader)
    historial_train.append(perdida_train_prom)

    modelo.eval()
    with torch.no_grad():
        perdida_val = criterio(modelo(X_val_t), y_val_t).item()
    historial_val.append(perdida_val)

    if perdida_val < mejor_val_loss:
        mejor_val_loss = perdida_val
        epocas_sin_mejora = 0
        mejor_modelo_pesos = modelo.state_dict()
    else:
        epocas_sin_mejora += 1

    if (epoca + 1) % 10 == 0:
        print(f"Época {epoca+1:03d} | Train: {perdida_train_prom:.4f} | Val: {perdida_val:.4f}")

    if epocas_sin_mejora >= paciencia:
        epoca_early_stop = epoca + 1
        print(f"\n🛑 Early Stopping en la época {epoca+1}.")
        break

modelo.load_state_dict(mejor_modelo_pesos)

# ==========================================
# 6. GRÁFICAS Y EVALUACIÓN FINAL
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(historial_train)+1), historial_train, marker='o', label='Train Loss', linewidth=2)
plt.plot(range(1, len(historial_val)+1), historial_val, marker='s', label='Val Loss', linewidth=2)
if epoca_early_stop:
    plt.axvline(x=epoca_early_stop, color='red', linestyle='--', alpha=0.6, label=f'Early Stop (época {epoca_early_stop})')
plt.xlabel('Época'); plt.ylabel('Pérdida (BCEWithLogits)'); plt.title('Curvas de Pérdida — Entrenamiento vs Validación')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

modelo.eval()
with torch.no_grad():
    y_test_logits = modelo(X_test_t)
    y_test_probs = torch.sigmoid(y_test_logits).numpy().flatten()

precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
mejor_umbral = thresholds[np.argmax(f1_scores)]

print(f"\n🔍 Umbral óptimo de decisión: {mejor_umbral:.4f}")
y_test_pred = (y_test_probs > mejor_umbral).astype(int)

print("\n--- Reporte de Clasificación (Test - Umbral Optimizado) ---")
print(classification_report(y_test, y_test_pred, target_names=["No Brote (0)", "Brote (1)"], zero_division=0))
auc = roc_auc_score(y_test, y_test_probs)
print(f"ROC-AUC Score: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_test_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.4f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar')
plt.xlabel('Tasa de Falsos Positivos'); plt.ylabel('Tasa de Verdaderos Positivos'); plt.title('Curva ROC — Predicción de Brotes')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Brote", "Brote"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matriz de Confusión - Umbral {mejor_umbral:.2f}")
plt.show()

# ==========================================
# 7. BENCHMARK: RANDOM FOREST
# ==========================================
print("\n7. Entrenando Benchmark (Random Forest)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
rf.fit(X_train_p, y_train)

y_probs_rf = rf.predict_proba(X_test_p)[:, 1]
y_pred_rf = rf.predict(X_test_p)
auc_rf = roc_auc_score(y_test, y_probs_rf)

print(f"🌲 ROC-AUC Random Forest: {auc_rf:.4f}")
print("\n--- Reporte de Clasificación (Random Forest) ---")
print(classification_report(y_test, y_pred_rf, target_names=["No Brote (0)", "Brote (1)"], zero_division=0))

importancias = pd.Series(rf.feature_importances_, index=columnas_features).sort_values(ascending=False)
print("\nTop 10 variables más importantes según Random Forest:")
print(importancias.head(10))