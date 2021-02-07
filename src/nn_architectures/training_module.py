import io
import os
import boto3
import requests
import pickle
import json
import joblib
import xgboost
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from datetime import timedelta
from xgboost import plot_importance


# Set environment variables
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
# dynamo ambos nombres
os.environ["dynamodbads"] =\
    ssm.get_parameter(Name='DynamoDBADS-develop',
                      WithDecryption=True)["Parameter"]["Value"]
os.environ["dynamodbadsv2"] =\
    ssm.get_parameter(Name='DynamoDBADS-develop',
                      WithDecryption=True)["Parameter"]["Value"]


def handler_loss_function(batch_size, penalization):

    if penalization:
        # Retorna la función de costos de cosmos con penalización
        def cosmos_loss_function(y_true, y_pred):
            # Covertir a tensor de tensorflow con keras como backend
            y_true = K.cast(y_true, dtype='float32')
            y_pred = K.cast(y_pred, dtype='float32')
            # Reshape como vector
            y_true = K.reshape(y_true, (-1, 1))
            y_pred = K.reshape(y_pred, (-1, 1))
            # Vector de error mae
            diff_error = y_pred - y_true
            # Cuenta el número de veces que se equivoca por abajo
            negative_values = K.cast(tf.math.count_nonzero(diff_error < 0),
                                     dtype='float32')
            size_train = K.shape(y_pred)[0]
            size_train = K.reshape(size_train, (-1, 1))
            loss = K.square(y_pred - y_true)
            loss = K.sum(loss, axis=1)
            loss = K.mean(loss)
            loss = loss * (0.1 + negative_values / batch_size)
            return loss

    elif penalization is False:
        # Retorna el error cuadratico medio
        def cosmos_loss_function(y_true, y_pred):
            y_true = K.cast(y_true, dtype='float32')
            y_pred = K.cast(y_pred, dtype='float32')
            y_true = K.reshape(y_true, (-1, 1))
            y_pred = K.reshape(y_pred, (-1, 1))
            size_train = K.shape(y_pred)[0]
            size_train = K.reshape(size_train, (-1, 1))
            loss = K.square(y_pred - y_true)
            loss = K.sum(loss, axis=1)
            loss = K.mean(loss)
            return loss
    return cosmos_loss_function


def distribucion_errores(y_pred, y_test):
    """
    Printear los errores por arriba y abajo del modelo
    Parameters
    ----------
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    diff = y_pred - y_test
    diff = np.reshape(diff, -1)
    negative_values = np.count_nonzero(diff < 0)
    dd = int(negative_values/y_pred.shape[0]*100) + 1
    aa = 100 - dd
    dd, aa = str(dd), str(aa)
    print("Porcentaje de errores por debajo:",
          negative_values/y_pred.shape[0]*100)
    print("Porcentaje de errores por arriba:",
          100 - negative_values/y_pred.shape[0]*100)
    return aa, dd


def training_history(history, epocas_hacia_atras, model_name, filename):

    # Hist training
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    # Funciones de costo
    loss_training = history.history['loss'][-epocas_hacia_atras:]
    loss_validation = history.history['val_loss'][-epocas_hacia_atras:]
    # Figura
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x_labels, loss_training, 'b', linewidth=2)
    ax.plot(x_labels, loss_validation, 'r', linewidth=2)
    ax.set_xlabel('Epochs', fontname="Arial", fontsize=14)
    ax.set_ylabel('Cosmos loss function', fontname="Arial", fontsize=14)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=20)
    ax.legend(['Training', 'Validation'], loc='upper left',
              prop={'size': 14})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    # Guardar_resultados del modelo en s3
    s3 = boto3.client("s3")
    bucket_train = os.environ["buckettrain"]
    fig.savefig(f"{model_name}.png")
    s3.upload_file(Filename=f"{model_name}.png",
                   Bucket=bucket_train,
                   Key=f"sagemaker_cosmos/resultados/{model_name}.png")

    return True


def plot_historial_entrenamiento(history, model_name, filename):
    """
    Según el historial de entrenamiento que hubo plotear el historial
    hacía atrás de las variables
    Parameters
    ----------
    history : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    size_training = len(history.history['val_loss'])
    fig = training_history(history, size_training, model_name,
                           filename + "_ultimas:" +
                           str(size_training) + "epocas")

    fig = training_history(history, int(1.5 * size_training / 2), model_name,
                           filename + "_ultimas:" +
                           str(1.5 * size_training / 2) + "epocas")

    fig = training_history(history, int(size_training / 2), model_name,
                           filename + "_ultimas:" + str(size_training / 2) +
                           "epocas")

    fig = training_history(history, int(size_training / 3), model_name,
                           filename + "_ultimas:" + str(size_training / 3) +
                           "epocas")
    fig = training_history(history, int(size_training / 4), model_name,
                           filename + "_ultimas:" + str(size_training / 4) +
                           "epocas")
    print(fig)


def read_pkl_s3(bucket, ruta):
    """
    La funcion lee un archivo pkl desde s3
    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo
    Returns
    -------
    data : Dataframe con los datos
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket,
                        Key=ruta)
    body = obj["Body"].read()
    data = pickle.loads(body)
    data.reset_index(inplace=True, drop=True)
    return data


def scaler_s3(bucket, nombre, ruta):
    """
    La funcion lee un archivo pkl desde s3
    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo
    Returns
    -------
    data : Dataframe con los datos
    """
    s3 = boto3.client("s3")
    s3.download_file(bucket, ruta, f"{nombre}")
    scaler = joblib.load(f"{nombre}.save")
    return scaler


def modelos_s3(bucket, nombre, ruta):
    """
    La funcion lee un archivo pkl desde s3
    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo
    Returns
    -------
    data : Dataframe con los datos
    """
    s3 = boto3.client("s3")
    print(bucket + ruta)
    s3.download_file(bucket, ruta, f"{nombre}.h5")
    model = load_model(f"{nombre}.h5", compile=False)
    return model


def load_ads(s3, fecha_actual, dias, ruta_s3):
    """
    La funcion permite traer los ads desde una fecha y una cantidad de
    dias hacia atras
    Parameters
    ----------
    s3 : cliente s3
    fecha_actual : fecha desde donde se realiza el analisis
    dias : int
        dias que ir a buscar hacia atras
    ruta_s3 : str
        prefijo con el que empieza la ruta
    Returns
    -------
    data_final : ads final
    """
    # Data final
    data_final = pd.DataFrame()
    bucket = os.environ["outputbucket"]
    # Calculamos los dias anteriores
    for i in range(dias):
        print(i)
        d = (fecha_actual-timedelta(i)).day
        m = (fecha_actual-timedelta(i)).month
        y = (fecha_actual-timedelta(i)).year
        prefijo = str(y) + "/" + str(m) + "/"
        print(prefijo)
        try:
            root = f"{ruta_s3}/{prefijo}ads_modelo_no_predict_{d}-{m}-{y}.pkl"
            data = read_pkl_s3(bucket, root)
            print(root)
            data_final = pd.concat([data_final, data])
        except Exception as e:
            print(e)
            pass

    return data_final


def analisis_colinealidades(df, target):
    """
    Visualizar las correlaciones de las características con la variable
    Cantidad
    Parameters
    ----------
    df : DataFrame
        ADS de cosmos
    Returns
    -------
    None.
    """
    for columna in df.columns:
        x1 = df[target]
        x2 = df[columna]
        fig, ax = plt.subplots(1, figsize=(22, 12))
        # qi = 0.1
        # qf = 0.9
        # min_x = x1.quantile(qi)
        # max_x = x1.quantile(qf)
        # min_y = x2.quantile(qi)
        # max_y = x2.quantile(qf)

        plt.scatter(x1, x2, color='orangered')
        titulo = f'Análisis de colinealidades variable: {columna}'
        plt.title(titulo, fontsize=30)
        plt.xlabel(target, fontsize=30)
        plt.ylabel(columna, fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=22)
        plt.legend([columna], fontsize=22,
                   loc="upper left")
        # plt.ylim(min_y, max_y)
        # plt.xlim(min_x, max_x)
        plt.show()


def post_message_to_slack(text, blocks=None):
    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': "xoxb-563621536019-1002633394961-7o6uK2375XZR2Y9sQgrbobHM",
        'channel': '#retrain-anglo',
        'text': text,
        'blocks': json.dumps(blocks) if blocks else None
    }).json()


def xgboost_processing(X_train, X_test, y_train, y_test, early_stopping_rounds,
                       columns_dataset):
    """
    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    early_stopping_rounds : TYPE
        DESCRIPTION.
    columns_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    y_pred : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    regressor : TYPE
        DESCRIPTION.

    """
    columns_X = columns_dataset
    # Reordenar en la forma (algo,)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    # XGBRegressor MODEL
    regressor = xgboost.XGBRegressor()
    # Entrenar el modelo
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  early_stopping_rounds=early_stopping_rounds)
    # Generar predicciones
    y_pred = regressor.predict(X_test)
    # sacar el peso de cada variable en el arbol de decisión
    feature_importance =\
        regressor.get_booster().get_score(importance_type="weight")
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())
    data = pd.DataFrame(
        data=values, index=keys,
        columns=["score"]).sort_values(by="score", ascending=False)
    list_feature_importance = []
    # sacar la importancia de las columnas
    for col, score in zip(columns_X, regressor.feature_importances_):
        list_feature_importance.append([col, score])
    data = pd.DataFrame(list_feature_importance,
                        columns=["feature", "importance_score"])
    data = data.sort_values(by="importance_score",
                            ascending=False).reset_index(drop=True)
    pickle.dump(regressor, open("xgboost.pickle.dat", "wb"))
    # Plot importancia de las variables
    plt.figure(figsize=(40, 20))
    plot_importance(regressor)
    plt.show()
    return y_pred, data, regressor


def correlation_selection(dataset, threshold):
    """
    Selecciona solo una de las columnas altamente correlacionadas

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (
                    corr_matrix.columns[j] not in col_corr):
                # getting the name of column
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    # deleting the column from the dataset
                    del dataset[colname]
    dataset = dataset.reset_index(drop=True)
    return dataset, dataset.columns.to_list()


def get_model_summary(model):
    """
    Retorna el sumary de los modelos

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    summary_string : TYPE
        DESCRIPTION.

    """

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
