# Paths.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Procesamiento de datos.
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")  # Se prepara la ruta.

print("Shape:", df.shape)
print("\nTipos:\n", df.dtypes)
print("\nNulos por columna (%):\n", (df.isna().mean() * 100).round(2).sort_values(ascending=False))
print("\nDuplicados:", df.duplicated().sum())
print("\nCardinalidad (nº valores únicos):\n", df.nunique().sort_values(ascending=False))

# Distribución del target. (Columna Importante)
target = "DEATH_EVENT"
print("\nDistribución del target:\n", df[target].value_counts(normalize=True).round(3))

# Fuga de información (Objeto / Columna "time")
df = df.drop(columns=["time"])

# Separación de X / y (Detallando los datos del analisis)
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# Split ANTES de preprocesar (Division entre Train, Validation and Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Columnas (Procesamiento de la red neuronal "Normalización")
num_cols = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium",
]
# Los binarios se dejan tal cual.
bin_cols = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("bin", "passthrough", bin_cols),
])

# Train = La red neuronal esta aprendiendo (Fit SOLO con train).
X_train_p = preprocesador.fit_transform(X_train)
# Validation and test, only tranformation.
X_val_p   = preprocesador.transform(X_val)
X_test_p  = preprocesador.transform(X_test)

# Verificación 
print("Shapes:", X_train_p.shape, X_val_p.shape, X_test_p.shape)
print("y_train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("y_val:  ", y_val.value_counts(normalize=True).round(3).to_dict())
print("y_test: ", y_test.value_counts(normalize=True).round(3).to_dict())

# Resultados ("(209, 11) (45, 11) (45, 11)")

## 209 filas para entrenar con 11 variables de entrada.

## 45 para validar y 45 para probar.

## Las 11 variables son las 6 continuas (escaladas) + las 5 binarias (pasadas directo).

## PRUEBA DE VALIDACIÓN DE DATOS.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Convertir a Tensores
# X: necesitamos float32 (decimales). y: necesitamos que sea una columna (N, 1)
X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_p, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Crear los DataLoaders (lotes de 32 en 32)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # shuffle=True mezcla los datos en cada época

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1) # Salida: 1 sola neurona para clasificación binaria
        )

    def forward(self, x):
        return self.net(x)

modelo = MLP(input_dim=11)