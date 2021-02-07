import joblib
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from src.nn_architectures.training_module import (handler_loss_function,
                                                  training_history)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

filename = "cnn_final_model"
# Preparamos la data
ads = pd.read_pickle("dataset/oversampling_ads.pkl")

features = ads.columns.to_list()
X = ads[features]

y = X[["Cantidad"]]
del X["Cantidad"]
columns_dataset = X.columns.to_list()

# Crear objecto scaler
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
# Standarizar
X = scaler.fit_transform(X)
# # Guardar objeto scaler
scaler_filename = f"scaler_dataset/{filename}_scaler.save"
joblib.dump(scaler, scaler_filename)
print(X.shape)
# transformar el feature vetor to matrix
X = np.reshape(X, (-1, 9, 3, 1))
print(X.shape)
# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=20)

# Si penalization es True, ocupa la función de costos que penaliza errores <0
penalization = False
# Elección de hiperparámetros según la penalización que se utiliza
if penalization == True:
    # Hiperparámetros de la red
    batch_size = 1024
    epochs = 1200
    # Hiperparámetros de los callbacks
    patience = 25
    min_delta = 500

elif penalization == False:
    # Hiperparámetros de la red
    batch_size = 1024
    epochs = 1200
    # Hiperparámetros de los callbacks
    patience = 25
    min_delta = 500

model_name = f"CNN: {filename}"
# define convolution model
model = Sequential()
model.add(Conv2D(32, input_shape=X_train.shape[1:],
                 kernel_size=(2, 1), padding="same", activation="relu"))
model.add(Conv2D(64, (2, 1), padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (2, 1), padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()

# Agregar las función de costo a keras
keras.losses.handler_loss_function = handler_loss_function
# Compile optimizer
model.compile(loss=handler_loss_function(batch_size, penalization),
              optimizer='nadam')
keras.callbacks.Callback()
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
                                           min_lr=5E-4)

callbacks = [stop_condition, learning_rate_schedule]

history = model.fit(X_train, y_train, validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=False,
                    verbose=1,
                    callbacks=callbacks)

size_training = len(history.history['val_loss'])
fig = training_history(
    history, size_training, model_name,
    filename + "_ultimas:" + str(size_training)+"epocas")
fig = training_history(
    history, int(1.5 * size_training / 2), model_name,
    filename + "_ultimas:" + str(1.5 * size_training / 2) + "epocas")
fig = training_history(
    history, int(size_training / 2), model_name,
    filename + "_ultimas:" + str(size_training / 2) + "epocas")
fig = training_history(
    history, int(size_training / 3), model_name,
    filename + "_ultimas:" + str(size_training / 3) + "epocas")
fig = training_history(
    history, int(size_training / 4), model_name,
    filename + "_ultimas:" + str(size_training / 4) + "epocas")


# Score del modelo entrenado
scores = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Mean squared error, Test:', scores)
# Predictions on y_test
y_pred = model.predict(X_test)
# Metricas de evaluación
mae_acum = abs(y_pred-y_test)
mae = int(mae_acum.mean())
std = int(mae_acum.std())
q95 = int(mae_acum.quantile(0.95))

print("=================================")
print("MAE -----> " + str(mae))
print("DEVEST --> " + str(std))
print("=================================")
# Save the model as .h5
# model.save(f"models_checkpoint/{filename}_{model_name}.h5")
diff = y_pred - y_test
diff = np.reshape(diff, -1)
negative_values = np.count_nonzero(diff < 0)
print("Porcentaje de errores por debajo:",
      negative_values/y_pred.shape[0]*100)


fig, ax = plt.subplots(1, figsize=(22, 12))
plt.scatter(y_test, y_pred, color='blue')
plt.scatter(y_test, y_pred, color='blue')
plt.scatter(y_test, y_test, color='red')
titulo = f'CNN oversampling + originales' +\
    f'| Data original: {y_test.shape[0]} filas' + '\n' +\
    f'MAE: {str(mae)} [lts] --- STD: {str(std)} [lts] --- Q95: {str(q95)} [lts]'
plt.title(titulo, fontsize=30)
plt.xlabel('Cantidades reales de combustible', fontsize=30)
plt.ylabel('Predicción CNN de combustible', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(["Predicciones", "Cantidades reales"], fontsize=30,
           loc="lower right")
plt.ylim(0, 4600)
plt.xlim(0, 4600)
plt.show()

model.save(f"models_checkpoint/{filename}_{model_name}.h5")
