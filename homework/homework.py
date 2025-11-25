# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def pregunta_01():
    # carga de datos
    df_x_tr = pd.read_pickle("files/grading/x_train.pkl")
    df_y_tr = pd.read_pickle("files/grading/y_train.pkl")
    df_x_te = pd.read_pickle("files/grading/x_test.pkl")
    df_y_te = pd.read_pickle("files/grading/y_test.pkl")

    # normaliza nombre de columna target
    if isinstance(df_y_tr, pd.DataFrame):
        if "default payment next month" in df_y_tr.columns:
            df_y_tr = df_y_tr.rename(columns={"default payment next month": "default"})
            df_y_tr = df_y_tr["default"]
    else:
        df_y_tr.name = "default"

    if isinstance(df_y_te, pd.DataFrame):
        if "default payment next month" in df_y_te.columns:
            df_y_te = df_y_te.rename(columns={"default payment next month": "default"})
            df_y_te = df_y_te["default"]
    else:
        df_y_te.name = "default"

    # quita columna target duplicada si aparece en X
    if "default payment next month" in df_x_tr.columns:
        df_x_tr = df_x_tr.drop(columns=["default payment next month"])
    if "default payment next month" in df_x_te.columns:
        df_x_te = df_x_te.drop(columns=["default payment next month"])

    # quita id
    if "ID" in df_x_tr.columns:
        df_x_tr = df_x_tr.drop(columns=["ID"])
    if "ID" in df_x_te.columns:
        df_x_te = df_x_te.drop(columns=["ID"])

    # corrige valores de education
    if "EDUCATION" in df_x_tr.columns:
        df_x_tr["EDUCATION"] = df_x_tr["EDUCATION"].apply(lambda n: 4 if n > 4 else n)
    if "EDUCATION" in df_x_te.columns:
        df_x_te["EDUCATION"] = df_x_te["EDUCATION"].apply(lambda n: 4 if n > 4 else n)

    # une y limpia nulos
    df_train = pd.concat([df_x_tr, df_y_tr], axis=1).dropna()
    df_test = pd.concat([df_x_te, df_y_te], axis=1).dropna()

    X_tr = df_train.drop(columns=["default"])
    y_tr = df_train["default"]

    X_te = df_test.drop(columns=["default"])
    y_te = df_test["default"]

    # columnas categóricas
    cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X_tr.columns]

    # preprocesamiento simple
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    # pipeline del modelo
    pipe = Pipeline(
        steps=[
            ("preprocess", transformer),
            ("model", RandomForestClassifier(random_state=58082)),
        ]
    )

    # hiperparámetros
    params = {
        "model__n_estimators": [500],
        "model__max_depth": [22],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1],
        "model__class_weight": [None],
    }

    # búsqueda de parámetros
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    # entrenar el modelo
    grid.fit(X_tr, y_tr)

    # guardar modelo
    model_dir = "files/models"
    model_path = os.path.join(model_dir, "model.pkl.gz")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with gzip.open(model_path, "wb") as f:
        pickle.dump(grid, f)

    os.makedirs("files/output", exist_ok=True)
    results = []

    # calcula métricas
    def compute_metrics(X, y, name_set):
        preds = grid.predict(X)
        return {
            "type": "metrics",
            "dataset": name_set,
            "precision": float(precision_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "recall": float(recall_score(y, preds)),
            "f1_score": float(f1_score(y, preds)),
        }

    results.append(compute_metrics(X_tr, y_tr, "train"))
    results.append(compute_metrics(X_te, y_te, "test"))

    # función para matriz de confusión
    def cm_to_dict(cm, name_set):
        return {
            "type": "cm_matrix",
            "dataset": name_set,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }

    # genera matrices de confusión
    cm_train = confusion_matrix(y_tr, grid.predict(X_tr))
    cm_test = confusion_matrix(y_te, grid.predict(X_te))

    results.append(cm_to_dict(cm_train, "train"))
    results.append(cm_to_dict(cm_test, "test"))

    # guarda métricas
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    pregunta_01()
