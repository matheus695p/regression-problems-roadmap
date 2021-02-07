import warnings
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from src.nn_architectures.training_module import (xgboost_processing,
                                                  correlation_selection)
warnings.filterwarnings("ignore")

# Preparamos la data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl")
X = ads

y = X[["Cantidad"]]
del X["Cantidad"]
columns_dataset = X.columns.to_list()
# Crear objecto scaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Normalizar
X = scaler.fit_transform(X)
# # Guardar objeto scaler
scaler_filename = "scaler_dataset/scaler_xgboost.save"
joblib.dump(scaler, scaler_filename)
# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=20)
early_stopping_rounds = 200
y_pred, feature_importance, xgboost = xgboost_processing(
    X_train, X_test, y_train, y_test, early_stopping_rounds, columns_dataset)
y_test = y_test.to_numpy()
y_test = np.reshape(y_test, (-1,))
mae_xgboost = mean_absolute_error(y_test, y_pred)
print(mae_xgboost)

# Feature selection
feature_selected = feature_importance["feature"].to_list()
dataset = ads[feature_selected]
# sacar correlación entre las columnas
correlation_columns =\
    dataset.corr().unstack().sort_values().\
    drop_duplicates().reset_index(drop=False)
correlation_columns.columns = ["var1", "var2", "correlation"]
correlation_columns = correlation_columns.sort_values(by=["correlation"],
                                                      ascending=False).\
    iloc[1:].reset_index(drop=True)
# columnas altamente correlacionadas
high_correlated_columns =\
    correlation_columns[(correlation_columns['correlation'] > 0.5) |
                        (correlation_columns['correlation'] < -0.5)]
threshold = 0.5
dataset_final, final_columns = correlation_selection(dataset, threshold)

total_ = feature_importance['importance_score'].sum()
feature_importance['importance_score'] =\
    feature_importance['importance_score']*100
feature_importance.to_csv('feature_importance.csv', index=False)
