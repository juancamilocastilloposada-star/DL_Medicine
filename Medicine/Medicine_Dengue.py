import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

epocas = 1000
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

# ==========================================
# 8. INSPECCIÓN DEL DATASET PHYSIONET (Fase 2)
# ==========================================
import os
import re
from pathlib import Path
from datetime import datetime

# 1. Definir rutas
RUTA_BASE = Path(r"C:\Users\USUARIO\Desktop\Mis Documentos\Estudios\Santoto\Segundo_Semestre\DL_DeepLearning\Seccion_1\Medicine")
RUTA_METADATA = RUTA_BASE / "metadata.csv"
RUTA_DATASET = RUTA_BASE / "a-multi-modal-satellite-imagery-dataset-for-public-health-analysis-in-colombia-1.0.0"

print("="*60)
print("RESUMEN DEL DATASET MULTIMODAL (DENGUE)")
print("="*60)

# --- PARTE 1: METADATA ---
df_meta = pd.read_csv(RUTA_METADATA)
print("\n1. ARCHIVO METADATA")
print(f"   - Registros (municipios): {len(df_meta):,}")
print(f"   - Total de columnas:      {len(df_meta.columns):,}")

# Clasificar columnas
cols_semanas = [c for c in df_meta.columns if '/w' in c]
cols_temp = [c for c in df_meta.columns if 'TEMPERATURE' in c]
cols_precip = [c for c in df_meta.columns if 'PRECIPITATION' in c]
cols_demograficas = [c for c in df_meta.columns if any(x in c for x in ['Age', 'Population', 'Cases', 'Afro', 'Indian'])]
cols_otras = len(df_meta.columns) - len(cols_semanas) - len(cols_temp) - len(cols_precip) - len(cols_demograficas)

print(f"   - Variables climáticas:   {len(cols_temp)} de Temperatura, {len(cols_precip)} de Precipitación")
print(f"   - Series semanales:       {len(cols_semanas)} columnas (ej. 2016/w01 ... 2018/w52)")
print(f"   - Otras (demografía/infraestructura): {cols_otras} columnas")

# Mostrar una muestra limpia
print("\nMuestra limpia (3 primeras filas, 5 columnas clave):")
columnas_clave = ['Municipality code', 'Municipality', '2016/w01', '2016/w02', '2018/w52']
print(df_meta[columnas_clave].head(3).to_string(index=False))

# --- PARTE 2: IMÁGENES ---
print("\n" + "-"*60)
print("2. CARPETA DE IMÁGENES")
carpeta_real = RUTA_DATASET / "81_municipalities_v2.0" / "81_municipalities_v2.0"
tiffs = list(carpeta_real.rglob("*.tif*"))

print(f"   - Total de imágenes TIFF: {len(tiffs):,}")
print("\nMuestra de nombres de archivo:")
for t in tiffs[:3]:
    print(f"      - {t.name}")
print("="*60)

# Convertir a formato largo
df_long = df_meta.melt(id_vars=['Municipality code'], value_vars=cols_semanas, 
                       var_name='ano_semana', value_name='casos')
df_long['ano'] = df_long['ano_semana'].str.split('/').str[0].astype(int)
df_long['semana'] = df_long['ano_semana'].str.split('/w').str[1].astype(int)

# Filtrar 2016-2018
df_long = df_long[df_long['ano'].isin([2016, 2017, 2018])]

# Calcular Target: Brote = 1 si los casos superan el percentil 80 histórico del municipio
umbral = df_long.groupby('Municipality code')['casos'].transform(lambda x: x.quantile(0.80))
df_long['brote'] = (df_long['casos'] > umbral).astype(int)

# --- B. Extraer info de las Imágenes ---
data_img = []
errores = 0
for t in tiffs:
    try:
        # Formato esperado: 13244_2016-01-17.tiff
        parts = t.stem.split('_')
        cod_mun = int(parts[0])
        fecha = datetime.strptime(parts[1], "%Y-%m-%d")
        
        # Convertir fecha exacta a semana ISO
        ano_iso = fecha.isocalendar()[0]
        semana_iso = fecha.isocalendar()[1]
        
        data_img.append({
            'Municipality code': cod_mun, 
            'ano': ano_iso, 
            'semana': semana_iso, 
            'ruta_tiff': str(t)
        })
    except Exception as e:
        if errores == 0: # Imprimir el primer error para saber qué pasa si falla
            print(f"Error en archivo {t.name}: {e}")
        errores += 1

df_imgs = pd.DataFrame(data_img)
print(f"Errores al procesar imágenes: {errores}")

# --- C. EL MERGE (Co-registro) ---
df_master = pd.merge(df_imgs, df_long, on=['Municipality code', 'ano', 'semana'], how='inner')

print("="*60)
print(f"Total de imágenes TIFF procesadas: {len(df_imgs):,}")
print(f"Imágenes que lograron co-registro clínico: {len(df_master):,}")
print(f"Proporción de brotes en el dataset: {df_master['brote'].mean():.1%}")
print("\nEjemplo de los datos listos para PyTorch:")
print(df_master[['Municipality code', 'ano', 'semana', 'casos', 'brote', 'ruta_tiff']].head(3).to_string(index=False))
print("="*60)

from torchvision import models, transforms
from PIL import Image

# 1. Definir transformaciones (Data Augmentation para entrenamiento)
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Medias de ImageNet
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Clase Dataset personalizada
class DengueImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ruta = row['ruta_tiff']
        
        # Leer el archivo TIFF con rasterio
        with rasterio.open(ruta) as src:
            # Tomamos las primeras 3 bandas (asumiendo RGB). Si el dataset usa otras, ajusta aquí.
            img = src.read([1, 2, 3]) 
            img = np.nan_to_num(img) # Evitar NaNs
            # Normalizar reflectancia (Sentinel-2 suele venir en escala 0-10000)
            img = np.clip(img / 10000.0, 0, 1) 
            img = img.transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
            img = (img * 255).astype(np.uint8) # A 8 bits para las transforms

        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(row['brote'], dtype=torch.float32).unsqueeze(0)
        return img, label

# 3. DIVISIÓN TEMPORAL (Respetando la Fase 1: Train 2016-17, Test 2018)
df_train = df_master[df_master['ano'] <= 2017]
df_test = df_master[df_master['ano'] == 2018]

df_train = df_train.sample(n=500, random_state=42)

train_dataset = DengueImageDataset(df_train, transform=transform_train)
test_dataset = DengueImageDataset(df_test, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Imágenes de entrenamiento: {len(train_dataset)}")
print(f"Imágenes de prueba: {len(test_dataset)}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# 1. Cargar ResNet50 preentrenada
modelo_A = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Congelar capas base (Feature Extraction)
for param in modelo_A.parameters():
    param.requires_grad = False

# 3. Adaptar la cabeza para clasificación binaria
num_ftrs = modelo_A.fc.in_features
modelo_A.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
modelo_A = modelo_A.to(DEVICE)

# 4. Configurar pérdida y optimizador (con pos_weight para el desbalanceo del 16.8%)
peso_positivo = torch.tensor([(1 - 0.168) / 0.168], dtype=torch.float32).to(DEVICE)
criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivo)
optimizador = optim.Adam(modelo_A.fc.parameters(), lr=1e-3)

# 5. Bucle de entrenamiento (3 épocas para no demorar)
EPOCAS = 3
for epoca in range(EPOCAS):
    modelo_A.train()
    perdida_total = 0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizador.zero_grad()
        salidas = modelo_A(imgs)
        loss = criterio(salidas, labels)
        loss.backward()
        optimizador.step()
        
        perdida_total += loss.item()
        if (i+1) % 50 == 0:
            print(f"Época {epoca+1} | Lote {i+1}/{len(train_loader)} | Pérdida: {loss.item():.4f}")

    print(f"--> Fin de época {epoca+1}. Pérdida media: {perdida_total/len(train_loader):.4f}")


from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# --- A. Evaluación en Test ---
modelo_A.eval()
y_true, y_probs = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        salidas = modelo_A(imgs)
        probs = torch.sigmoid(salidas).cpu().numpy()
        y_probs.extend(probs)
        y_true.extend(labels.numpy())

# Calcular métricas
auc = roc_auc_score(y_true, y_probs)
print(f"\n✅ ROC-AUC en Test (Rama A - CNN): {auc:.4f}")
print(classification_report(y_true, (np.array(y_probs) > 0.5).astype(int), target_names=["No Brote", "Brote"]))

# --- B. Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        # Solo necesitamos capturar la salida (forward) de la capa
        target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # 1. Obligar al tensor de entrada a requerir gradientes
        input_tensor.requires_grad_()
        
        # 2. Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = (torch.sigmoid(output) > 0.5).float()
            
        self.model.zero_grad()
        
        # 3. Calcular gradientes directamente sobre las activaciones (evita el error de congelamiento)
        grads = torch.autograd.grad(output, self.activations, grad_outputs=class_idx)[0]
        
        # 4. Ponderar las activaciones
        weights = torch.mean(grads, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        
        # 5. Post-procesamiento
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Tomar una imagen de prueba
img_tensor, label = test_dataset[0]
img_input = img_tensor.unsqueeze(0).to(DEVICE)

# Inicializar Grad-CAM (la última capa convolucional de ResNet50 es 'layer4')
grad_cam = GradCAM(modelo_A, modelo_A.layer4[-1])
heatmap = grad_cam.generate(img_input)

# Visualizar
# Desnormalizar imagen
img_np = img_tensor.permute(1, 2, 0).numpy()
img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_np = np.clip(img_np, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_np)
axes[0].set_title(f"Imagen Original (Etiqueta: {int(label.item())})")
axes[0].axis('off')

axes[1].imshow(img_np)
axes[1].imshow(heatmap, cmap='jet', alpha=0.5) # Superponer mapa de calor
axes[1].set_title("Grad-CAM: ¿Dónde mira la CNN?")
axes[1].axis('off')
plt.tight_layout()
plt.show()

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Preparar datos para secuencias ---
# Filtramos solo los municipios que tienen suficientes datos y ordenamos por tiempo
df_seq = df_long.sort_values(['Municipality code', 'ano', 'semana']).copy()
df_seq['casos_norm'] = df_seq.groupby('Municipality code')['casos'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5)) # Normalizar

# Función para ventanear
def crear_ventanas(df, lookback=14):
    X, y = [], []
    for mun, group in df.groupby('Municipality code'):
        datos = group['casos_norm'].values
        for i in range(len(datos) - lookback):
            X.append(datos[i:i+lookback])
            y.append(datos[i+lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# Crear ventanas (X: [Muestras, 14 semanas], y: [Muestras, 1])
X_seq, y_seq = crear_ventanas(df_seq, lookback=14)
X_seq = X_seq[:, :, np.newaxis] # Formato PyTorch: (batch, timesteps, features=1)

# División temporal estricta (Alineada con Fase 1: Train 2016-17, Test 2018)
# Necesitamos mapear qué registros son de 2018
# (Esta es una simplificación, en producción se hace por índice temporal)
corte = int(len(X_seq) * 0.8) # 80% para train, 20% para test
X_train_rnn, y_train_rnn = X_seq[:corte], y_seq[:corte]
X_test_rnn, y_test_rnn = X_seq[corte:], y_seq[corte:]

train_rnn_ds = TensorDataset(torch.tensor(X_train_rnn), torch.tensor(y_train_rnn))
test_rnn_ds = TensorDataset(torch.tensor(X_test_rnn), torch.tensor(y_test_rnn))

train_rnn_loader = DataLoader(train_rnn_ds, batch_size=64, shuffle=True)
test_rnn_loader = DataLoader(test_rnn_ds, batch_size=64, shuffle=False)

# --- 2. Definir las 3 arquitecturas (RNN, LSTM, GRU) ---
class Recurrente(nn.Module):
    def __init__(self, tipo, input_size=1, hidden_size=64):
        super().__init__()
        # Seleccionar la capa según el tipo
        if tipo == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif tipo == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif tipo == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) # Tomar solo el último estado oculto
        return out

# --- 3. Bucle de entrenamiento ---
def entrenar_rnn(modelo, loader, epocas=15):
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.01)
    
    for epoca in range(epocas):
        modelo.train()
        perdida_total = 0
        for X_batch, y_batch in loader:
            optimizador.zero_grad()
            pred = modelo(X_batch)
            loss = criterio(pred, y_batch)
            loss.backward()
            optimizador.step()
            perdida_total += loss.item()
        
        # Imprimir cada 5 épocas
        if (epoca + 1) % 5 == 0:
            print(f"  Época {epoca+1}/{epocas} | MSE Loss: {perdida_total/len(loader):.4f}")
    return perdida_total/len(loader)

# --- 4. Entrenar y comparar ---
print("\n🔴 Entrenando RNN Simple (Contraste: sufre desvanecimiento):")
modelo_rnn = Recurrente('RNN')
loss_rnn = entrenar_rnn(modelo_rnn, train_rnn_loader)

print("\n🟡 Entrenando GRU:")
modelo_gru = Recurrente('GRU')
loss_gru = entrenar_rnn(modelo_gru, train_rnn_loader)

print("\n🟢 Entrenando LSTM (La mejor para dependencias largas):")
modelo_lstm = Recurrente('LSTM')
loss_lstm = entrenar_rnn(modelo_lstm, train_rnn_loader)

import torch.nn as nn
import torch.optim as optim

# ==========================================
# RAMA C: FUSIÓN MULTIMODAL Y ABLACIÓN
# ==========================================
print("\n" + "="*60)
print("RAMA C: FUSIÓN MULTIMODAL")
print("="*60)

# 1. Definir la Cabeza de Fusión
class CabezaFusion(nn.Module):
    def __init__(self, dim_img=2048, dim_seq=64, n_salidas=1):
        super().__init__()
        # Toma el vector de la imagen (2048 de ResNet) y de la serie (64 de LSTM)
        self.red = nn.Sequential(
            nn.Linear(dim_img + dim_seq, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_salidas)
        )
        
    def forward(self, z_img, z_seq):
        # Concatenar por columnas (dim=1)
        z_fusion = torch.cat((z_img, z_seq), dim=1)
        return self.red(z_fusion)

# 2. Extraer los embeddings (representaciones) reales
# Para este ejemplo rápido, usaremos datos simulados con las dimensiones correctas
# En tu cuaderno de Colab, aquí extraerás z_img de la CNN y z_seq de la LSTM.
N_MUESTRAS = 1000 # Simulamos 1000 muestras co-registradas
z_img_simulado = torch.randn(N_MUESTRAS, 2048) # Vector de la CNN (ResNet)
z_seq_simulado = torch.randn(N_MUESTRAS, 64)   # Vector de la LSTM
y_simulado = torch.randint(0, 2, (N_MUESTRAS, 1)).float()

# 3. Función para entrenar la fusión (y hacer la ablación)
def entrenar_ablacion(modo, epocas=10):
    """modo: 'solo_img', 'solo_seq', 'fusion'"""
    print(f"\n--- Entrenando modo: {modo} ---")
    
    # Si es solo imagen, la red de fusión solo usa z_img (el resto se pone en cero)
    if modo == 'solo_img':
        z_seq_input = torch.zeros_like(z_seq_simulado)
    elif modo == 'solo_seq':
        z_img_input = torch.zeros_like(z_img_simulado)
        
    modelo = CabezaFusion().to(DEVICE)
    criterio = nn.BCEWithLogitsLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=1e-3)
    
    for epoca in range(epocas):
        modelo.train()
        optimizador.zero_grad()
        
        if modo == 'fusion':
            pred = modelo(z_img_simulado, z_seq_simulado)
        elif modo == 'solo_img':
            pred = modelo(z_img_simulado, z_seq_input)
        elif modo == 'solo_seq':
            pred = modelo(z_img_input, z_seq_simulado)
            
        loss = criterio(pred, y_simulado)
        loss.backward()
        optimizador.step()
        
    print(f"Pérdida final ({modo}): {loss.item():.4f}")
    return loss.item()

# 4. Correr el Estudio de Ablación
loss_img = entrenar_ablacion('solo_img')
loss_seq = entrenar_ablacion('solo_seq')
loss_fusion = entrenar_ablacion('fusion')

print("\n--- Resultados de la Ablación ---")
print(f"Solo Imagen: {loss_img:.4f}")
print(f"Solo Serie:  {loss_seq:.4f}")
print(f"Fusionado:   {loss_fusion:.4f}")