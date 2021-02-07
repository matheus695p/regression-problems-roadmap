import os
import sys
import boto3
import joblib
import keras
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from src.nn_architectures.training_module import (handler_loss_function,
                                                  plot_historial_entrenamiento,
                                                  distribucion_errores,
                                                  read_pkl_s3,
                                                  scaler_s3, modelos_s3,
                                                  post_message_to_slack)
import warnings
warnings.filterwarnings("ignore")

# manejador de credenciales
ssm = boto3.client("ssm")
# nombre de la mina
mine = ssm.get_parameter(Name='mine',
                         WithDecryption=True)["Parameter"]["Value"]
# bucket de entrada
os.environ["inputbucket"] =\
    ssm.get_parameter(Name='InputBucket-develop',
                      WithDecryption=True)["Parameter"]["Value"]
# bucket de entrenamiento
os.environ["buckettrain"] =\
    ssm.get_parameter(Name='TrainBucket-develop',
                      WithDecryption=True)["Parameter"]["Value"]
# bucket de salida
os.environ["outputbucket"] =\
    ssm.get_parameter(Name='OutputBucket-develop',
                      WithDecryption=True)["Parameter"]["Value"]
print(os.path.abspath(sys.argv[0]))


def train_nn_cosmos(filename, columns, model, hyperparameters={
        "penalization": True,
        "batch_size": 4096,
        "epochs": 800,
        "patience": 70,
        "min_delta": 400,
        "optimizer": "adam",
        "lr_factor": 0.5,
        "lr_patience": 25,
        "lr_min": 5E-4,
        "validation_size": 0.2}):
    """
    Entrenar con red neuronal en función de las columnas que se le entregan
    para poder entrenear modularizadamente los distintos modelos

    Parameters
    ----------
    filename : string
        nombre del modelo.

    columns : string
        DESCRIPTION.

    model : object
        arquitectura definida en el main.py de este archivo.
        Ej:

        model = Sequential()
        model.add(Dense(2048, input_dim=X_train.shape[1], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear'))
        model.summary()

    hyperparameters : Dict
        DESCRIPTION. The default is:
        hyperparameters = {
            "penalization": True,
            "batch_size": 4096,
            "epochs": 800,
            "patience": 70,
            "min_delta": 400,
            "optimizer": "adam",
            "lr_factor": 0.5,
            "lr_patience": 25,
            "lr_min": 5E-4,
            "validation_size": 0.2
            }
    """
    # hiperpametros para el entrenamiento del modelo
    # si es True, ejecutará con una función de costos personalizada
    penalization = hyperparameters["penalization"]
    # tamaño del lote de entrenamiento
    batch_size = hyperparameters["penalization"]
    # épocas de entrenamiento
    epochs = hyperparameters["penalization"]
    # paciencia del earlystopping
    patience = hyperparameters["penalization"]
    # lo mínimo que tiene que bajar el earlystopping para terminar el train
    min_delta = hyperparameters["penalization"]
    # optimizador
    optimizer = hyperparameters["optimizer"]
    # factor de división learining rate
    lr_factor = hyperparameters["lr_factor"]
    # paciencia del learning rate
    lr_patience = hyperparameters["lr_patience"]
    # mínima tasas de aprendizaje que puede llegar a tener
    lr_min = hyperparameters["lr_min"]
    # porcentaje de los datos a validación
    validation_size = hyperparameters["validation_size"]

    # llamar a s3
    s3 = boto3.client("s3")
    # Nombre del modelo en la nube
    bucket_train = os.environ["buckettrain"]
    num_classes = 0
    print(filename)
    # Data eqtiquetada
    path_labelled = "sagemaker_cosmos/preprocessed_ads_window.pkl"
    data_oversampling = read_pkl_s3(bucket_train, path_labelled)
    data_oversampling = data_oversampling[columns].reset_index(drop=True)
    # Data original
    path_original = "sagemaker_cosmos/preprocessed_ads.pkl"
    data_original = read_pkl_s3(bucket_train, path_original)
    data_original = data_original[columns].reset_index(drop=True)
    # ADS con la data generada por los snapshots y
    # la original de final de ciclo
    ads = pd.concat([data_original, data_oversampling], axis=0)
    ads.reset_index(drop=True, inplace=True)
    # features que se utilizan en el entrenamiento
    features = ads.columns.to_list()
    X = ads[features]
    # Separar variable target, de las features
    y = X[["cantidad"]]
    y_original = data_original[["cantidad"]]
    del X["cantidad"], data_original["cantidad"]
    columns_dataset = X.columns.to_list()
    # scaler para normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    # normalizar
    X = scaler.fit_transform(X)
    X_original = scaler.transform(data_original)
    # guardar el objeto en el bucket de train con key pruebas de entrenamiento
    scaler_filename = f"{filename}.save"
    joblib.dump(scaler, scaler_filename)
    # Dividir los conjuntos de datos
    X = pd.DataFrame(X, columns=columns_dataset).reset_index(drop=True)
    X_original = pd.DataFrame(
        X_original, columns=columns_dataset).reset_index(drop=True)
    y_original = y_original.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Semilla de alatoredad
    alpha = 20
    # Dividir los conjuntos de datos
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=alpha)
    original = pd.concat([y_original, X_original], axis=1)
    test = pd.concat([y_test, X_test], axis=1)
    alpha = original.merge(test, how="inner")
    # Pasar los datos a numpy array
    X_test = X_test.to_numpy(dtype="float64")
    y_test = y_test.to_numpy(dtype="float64")
    X_train = X_train.to_numpy(dtype="float64")
    y_train = y_train.to_numpy(dtype="float64")
    # definir el nombre del modelo
    model_name = f"{filename}"
    # empezar el proceso de entrenamiento del modelo
    model.summary()
    # Agregar las función de costo a keras
    keras.losses.handler_loss_function = handler_loss_function
    # Compile optimizer
    model.compile(loss=handler_loss_function(batch_size, penalization),
                  optimizer=optimizer)
    keras.callbacks.Callback()
    stop_condition = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   mode='min',
                                                   patience=patience,
                                                   verbose=1,
                                                   min_delta=min_delta,
                                                   restore_best_weights=True)
    learning_rate_schedule = ReduceLROnPlateau(monitor="val_loss",
                                               factor=lr_factor,
                                               patience=lr_patience,
                                               verbose=1,
                                               mode="auto",
                                               cooldown=0,
                                               min_lr=lr_min)
    callbacks = [stop_condition, learning_rate_schedule]
    # sacar el historial de entrenamiento
    history = model.fit(X_train, y_train, validation_split=validation_size,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=False,
                        verbose=1,
                        callbacks=callbacks)
    # Plotear los resultados del entrenamiento
    plot_historial_entrenamiento(history, model_name, filename)
    # Guardar el modelo actualmente reentrenado en la otra ruta
    model.save(f"{filename}.h5")
    s3.upload_file(Filename=f"{filename}.h5",
                   Bucket=bucket_train,
                   Key=f"sagemaker_cosmos/modelos/{filename}.h5")
    # Score del modelo entrenado
    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Mean squared error, Test:', scores)
    # Solo sacar las columnas originales
    y_test_original = alpha[["cantidad"]].to_numpy()
    x_test_original = alpha[columns_dataset]
    # Predicciones en conjunto de testeo
    y_pred = model.predict(X_test)
    # Predictions on y_test
    y_pred_original = model.predict(x_test_original)
    # Metricas de evaluación
    mae_acum = abs(y_pred_original-y_test_original)
    mae = round(mae_acum.mean(), 3)
    std = round(mae_acum.std(), 3)
    q95 = round(pd.DataFrame(mae_acum).quantile(0.95)[0], 3)
    aa, dd = distribucion_errores(y_pred_original, y_test_original)
    print(f"Los errores por abajo son: {dd} %")
    print(f"Los errores por arriba son: {aa} %")
    max_original = max(y_original.max())
    print(max_original)
    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(y_test, y_pred, color='blue')
    plt.scatter(y_test_original, y_pred_original, color='green')
    plt.scatter(y_test_original, y_test_original, color='red')
    titulo = 'NN oversampling + originales' +\
        f'| Data original: {alpha.shape[0]} filas' + '\n' +\
        f'MAE: {str(mae)} [lts]' +\
        f'--- STD: {str(std)} [lts] --- Q95: {str(q95)} [lts]' + '\n' +\
        f'errores por arriba/errores por abajo : {aa}' + '/' + f'{dd}'
    plt.title(titulo, fontsize=30)
    plt.xlabel('Cantidades reales de combustible', fontsize=30)
    plt.ylabel('Predicción NN de combustible', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(
        ["Predicciones oversampling",
         "Predicciones que estan en el conjunto de test y data original",
         "Cantidades reales", ""], fontsize=22, loc="lower right")
    plt.ylim(0, 4600)
    plt.xlim(0, 4600)
    plt.show()
    # resultados del modelo
    resultados_name = "resultados_" + filename
    fig.savefig(f"{resultados_name}.png")
    s3.upload_file(Filename=f"{resultados_name}.png",
                   Bucket=bucket_train,
                   Key=f"sagemaker_cosmos/resultados/{resultados_name}.png")

    y_test_original = pd.DataFrame(y_test_original, columns=["Cantidad"])
    y_pred_original = pd.DataFrame(y_pred_original, columns=["predicciones"])
    mae_acum = pd.DataFrame(mae_acum, columns=["error"])
    data = pd.concat([y_test_original, y_pred_original, mae_acum], axis=1)

    df_count =\
        data['error'].value_counts(bins=15).reset_index(drop=False)
    df_count.columns = ["rango", "frecuencia errores"]
    df_count['Porcentaje'] =\
        df_count['frecuencia errores'] / \
        df_count['frecuencia errores'].sum() * 100

    print("===========================================================")
    print("Errores en el conjunto de testeo original:")
    print("MAE -----> " + str(mae))
    print("DEVEST --> " + str(std))
    print("QUANTILE 0.95--> " + str(q95))
    print("El número de clases utilizado fue:", num_classes)
    print("Rangos cantidad de puntos en los que los errores sobre 500 lts:")
    print(df_count)
    print("===========================================================")

    # Cargar el modelo actual en productivo
    path_modelo_actual =\
        f"sagemaker_cosmos/productivo/modelos_productivos/{filename}.h5"
    path_scaler_actual =\
        f"sagemaker_cosmos/productivo/scalers_productivos/{filename}.save"
    # Cargar el scaler del que esta productivo actualmente
    scaler_productivo = scaler_s3(bucket_train, filename, path_scaler_actual)

    # Normalizar con los scalers del modelo actual
    x_test_original = scaler.inverse_transform(x_test_original)
    x_test = scaler_productivo.transform(x_test_original)
    # Cargar modelo actual en predictivo
    modelo_antiguio = modelos_s3(bucket_train, filename, path_modelo_actual)
    y_pred_antiguo = modelo_antiguio.predict(x_test)
    # en esta parte deberia leer el archivo csv
    ruta = 'sagemaker_cosmos/mae_modelos/mae_modelos.pkl'
    df_mae = read_pkl_s3(bucket_train, ruta)
    mae_antiguo = df_mae[filename].iloc[0]

    # Condiciones de deploy
    if filename == "modelo_global":
        limite = 250
    elif filename == "modelo_gps_loads_specto":
        limite = 280
    elif filename == "modelo_gps_mems_specto":
        limite = 300
    elif filename == "modelo_loads_mems_specto":
        limite = 300
    elif filename == "modelo_gps_loads_mems":
        limite = 300
    elif filename == "modelo_gps_specto":
        limite = 300
    elif filename == "modelo_mems_specto":
        limite = 300
    elif filename == "modelo_loads_mems":
        limite = 300
    elif filename == "modelo_gps_mems":
        limite = 300
    elif filename == "modelo_gps_loads":
        limite = 350
    elif filename == "modelo_gps":
        limite = 350
    elif filename == "modelo_loads":
        limite = 350
    elif filename == "modelo_mems":
        limite = 350
    elif filename == "modelo_specto":
        limite = 350

    if mae >= limite:
        space = "=============================================="
        space = space + space
        msg = f"({mine}) El proceso de re-entrenamiento del {filename} finalizó. " + '\n' +\
            f"El modelo {filename} no superó la condición límite de {limite}" + '\n' +\
            " litros de error, No habrá deploy de un nuevo modelo"+'\n' +\
            f" modelo en actual: {mae_antiguo} litros de error"+'\n' +\
            f" modelo recien entrenado: {mae} litros de error"
        texto = space + '\n' + msg + '\n' + space
        post_message_to_slack(texto)
        pass
    elif mae < limite:
        # Condiciones de subir o no un modelo
        if mae >= mae_antiguo:
            print("==========================================")
            print(f"Es mejor el modelo antiguo: {filename}")
            print("==========================================")
            space = "=============================================="
            space = space + space
            msg = f"({mine}) El proceso de re-entrenamiento del {filename} finalizó. " + '\n' +\
                f"El mae del modelo anterior fue de: mae = {mae_antiguo}" + '\n' +\
                f"El mae del modelo actual es de: mae ={mae}" + '\n' + \
                f" El modelo anterior es mejor que el actual, " + \
                "no hay cambio de modelo"
            texto = space + '\n' + msg + '\n' + space
            post_message_to_slack(texto)

        elif mae < mae_antiguo:
            print("==========================================")
            print(f"Es mejor el modelo nuevo: {filename}")
            print("==========================================")
            # Guardar el modelo nuevo en productivo
            model.save(f"{filename}.h5")
            s3.upload_file(Filename=f"{filename}.h5",
                           Bucket=bucket_train,
                           Key=path_modelo_actual)
            # Guardar el scaler en productivo
            scaler_filename = f"{filename}.save"
            joblib.dump(scaler, scaler_filename)
            s3.upload_file(Filename=f"{filename}.save",
                           Bucket=bucket_train,
                           Key=path_scaler_actual)
            # guardar mae en data frame
            df_mae.at[0, filename] = mae
            space = "=============================================="
            space = space + space
            msg = f"({mine}) El proceso de re-entrenamiento del {filename} finalizó. " + '\n' +\
                f"El mae del modelo anterior fue de: mae = {mae_antiguo}" + '\n' +\
                f"El mae del modelo actual es de: mae ={mae}" + '\n' + \
                f" El modelo actual es mejor que el anterior, " + \
                "se cambia el modelo a producción" + '\n' +\
                f" dejandolo en la ruta {path_modelo_actual} " + '\n' +\
                f"y el scaler en {path_scaler_actual}"
            texto = space + '\n' + msg + '\n' + space
            post_message_to_slack(texto)
    # subir mae a archivo pkl
    df_mae.to_pickle('mae_modelos.pkl')
    s3.upload_file(Filename="mae_modelos.pkl",
                   Bucket=bucket_train,
                   Key=ruta)


def verificar_a_en_b(lista1, lista2):
    """
    Verificar que todos los elementos de la lista1 estan en la lista2

    Parameters
    ----------
    lista1 : list
        Lista con combinaciones.
    lista2 : list
        Lista con todas las fuentes.

    Returns
    -------
    is_in : int
        Retorna 1 si todos los elemento estan en la lista1 estan en la lista2.

    """
    is_in = 1
    for elemento in lista1:
        if elemento in lista2:
            is_in = is_in*1
        else:
            is_in = is_in*0
    return is_in


def obtener_nombre_modelo(lista_fuentes):
    """
    Obtener nombre del modelo de acuerdo a las fuentes

    Parameters
    ----------
    lista_fuentes : list
        Listado de fuentes.

    Returns
    -------
    nombre_modelo : str
        Nombre del modelo con fuentes.

    """
    nombre_modelo = 'modelo'
    if 'gps' in lista_fuentes:
        nombre_modelo = nombre_modelo + '_' + 'gps'
    if 'loads' in lista_fuentes:
        nombre_modelo = nombre_modelo + '_' + 'loads'
    if 'mems' in lista_fuentes:
        nombre_modelo = nombre_modelo + '_' + 'mems'
    if 'specto' in lista_fuentes:
        nombre_modelo = nombre_modelo + '_' + 'specto'
    return nombre_modelo


def reordenar_fuentes(lista_fuentes):
    """
    Reoordenar fuentes

    Parameters
    ----------
    lista_fuentes : list
        Listado de fuentes.

    Returns
    -------
    nuevo_listado : list
        Listado de fuentes ordenado.

    """
    # nuevo listado reoordendano
    nuevo_listado = []
    if 'gps' in lista_fuentes:
        nuevo_listado.append('gps')
    if 'loads' in lista_fuentes:
        nuevo_listado.append('loads')
    if 'mems' in lista_fuentes:
        nuevo_listado.append('mems')
    if 'specto' in lista_fuentes:
        nuevo_listado.append('specto')
    return nuevo_listado


def columnas_fuentes(lista_fuentes):
    """
    Columnas asociadas a cada una de las fuentes

    Parameters
    ----------
    lista_fuentes : list(str)
        Listado de fuentes.

    Returns
    -------
    columnas : list(str)
        Listado de columnas asociada a cada una de las fuentes.

    """
    loads_columns = ['cantidad', 'cantidad_anterior', 'suma_toneladas',
                     'diff_hrs', 'carga_normal', 'carga_normal_anterior',
                     't_cargado', 'numero_cargas', 'tiempo_ciclo_mean']
    gps_columns = ['cantidad', 'cantidad_anterior', 't_final',
                   'distancia_gps', 'diff_hrs', 'num_registros',
                   'diff_cota_sub', 'diff_cota', 'tiempo', 't_apagado',
                   'carga_normal', 'carga_normal_anterior', 'vel_mean', 'vel025',
                   'vel075', 'distancia_pendiente']
    mems_columns = ['cantidad', 'cantidad_anterior', 'presion_total',
                    'presion_media', 'temp_neumaticos', 't_cargado_mems',
                    't_encendido_mems']
    specto_columns = ['cantidad', 'cantidad_anterior', 'prediccion_fuel_rate',
                      'rpm']
    columnas = []
    if 'gps' in lista_fuentes:
        for columna in gps_columns:
            if columna not in columnas:
                columnas.append(columna)
    if 'loads' in lista_fuentes:
        for columna in loads_columns:
            if columna not in columnas:
                columnas.append(columna)
    if 'mems' in lista_fuentes:
        for columna in mems_columns:
            if columna not in columnas:
                columnas.append(columna)
    if 'specto' in lista_fuentes:
        for columna in specto_columns:
            if columna not in columnas:
                columnas.append(columna)
    return columnas


def modelo_gps_loads_mems_specto(n_features):
    """
    Modelo para fuentes gps, loads, mems y specto

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_loads_specto(n_features):
    """
    Modelo para fuentes gps, loads y specto

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Objeto de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_mems_specto(n_features):
    """
    Modelo gps, mems y specto.

    Parameters
    ----------
    n_features : int
        Numero de features.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_loads_mems_specto(n_features):
    """
    Modelo de loads, mems y specto.

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_loads_mems(n_features):
    """
    Modelo para gps, loads y mems

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_loads(n_features):
    """
    Modelo gps y loads.

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_mems(n_features):
    """
    Modelo gps y mems

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_loads_mems(n_features):
    """
    Modelo de loads y mems

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_loads_specto(n_features):
    """
    Modelo de loads y specto

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_mems_specto(n_features):
    """
    Modelo mems y specto

    Parameters
    ----------
    n_features : int
        Numero de caracterisitcas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps_specto(n_features):
    """
    Modelo gps y specto

    Parameters
    ----------
    n_features : int
        Numero de caracterisitcas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_gps(n_features):
    """
    Modelo de gps

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_loads(n_features):
    """
    Modelo loads

    Parameters
    ----------
    n_features : int
        Numero de caracteristicas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_mems(n_features):
    """
    Modelo de mems
    Parameters
    ----------
    n_features : int
        Numero de carcateristicas.

    Returns
    -------
    model : object
        Objeto de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def modelo_specto(n_features):
    """
    Modelo de specto

    Parameters
    ----------
    n_features : int
        Numero de caracterisitcas.

    Returns
    -------
    model : object
        Modelo de keras.

    """
    model = Sequential()
    model.add(Dense(2048, input_dim=n_features, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model
