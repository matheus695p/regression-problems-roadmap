import warnings
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from kerastuner.tuners import RandomSearch
from fine_tuning_mudule import HyperModel
from src.nn_architectures.training_module import get_model_summary
warnings.filterwarnings("ignore")


filename = "Final_model"
# carga data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl").reset_index(drop=True)
dataset = ads

features = ads.columns.to_list()
X = ads[features]
y = X[["Cantidad"]]
del X["Cantidad"]
columns_dataset = X.columns.to_list()
# Data preparation
scaler = StandardScaler()
# Normalizar
X = scaler.fit_transform(X)
# # Guardar objeto scaler
scaler_filename = f"scaler_dataset/{filename}_scaler.save"
joblib.dump(scaler, scaler_filename)

# Dividir los conjuntos de datos
X = pd.DataFrame(X, columns=columns_dataset).reset_index(drop=True)
y = y.reset_index(drop=True)
X = X.to_numpy(dtype="float64")
y = y.to_numpy(dtype="float64")
# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=20)
# hiperparametros del finetuning
filename = "Fine_tunning_model"
batch_size = 60
epochs = 150
patience = 25
min_delta = 500
input_shape = (X_train.shape[1],)
# modelo parametrizado
hypermodel = HyperModel(input_shape)
# tunner de búsqueda del modelo
tuner_rs = RandomSearch(hypermodel, objective='mse', seed=20,
                        max_trials=150,
                        executions_per_trial=3,
                        directory='fine_tuning/')
tuner_rs.search_space_summary()
keras.callbacks.Callback()
# stop conditions
stop_condition = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               patience=patience,
                                               verbose=1,
                                               min_delta=min_delta,
                                               restore_best_weights=True)

learning_rate_schedule = ReduceLROnPlateau(monitor="val_loss",
                                           factor=0.5,
                                           patience=25,
                                           verbose=1,
                                           mode="auto",
                                           cooldown=0,
                                           min_lr=5E-3)

callbacks = [stop_condition, learning_rate_schedule]


tuner_rs.search(X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1)

# Se guarda en un txt las 10 mejores arquitecturas
models = tuner_rs.get_best_models(num_models=10)

idx = 0
with open('fine_tuning/Best_Architectures.txt', 'w') as ff:
    for model in models:
        ss = get_model_summary(model)
        ff.write('\n')
        ff.write(ss)
        ff.write(str(model.get_config()))


best_model = tuner_rs.get_best_models(num_models=1)[0]
loss, mse = best_model.evaluate(X_test, y_test)
print(best_model.summary())

model_json = best_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("fine_tuning_model.h5")
print("Se guardo el mejor modelo de la búsqueda")

# Guardar el json para saber la arquitectura
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos
loaded_model.load_weights("fine_tuning_model.h5")
print("Modelo cargado")
# Evaluar el modelo
loaded_model.compile(loss='mse', optimizer='adam')
score = loaded_model.evaluate(X_test, y_test, verbose=1)
print("El error del modelo es:", score)
