import os
import sys
import pytz
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.preprocessing.labeller_module import (ads_s3_filtros, timezone_fechas,
                                               sacar_cargas_anteriores, read_pkl_s3)
from src.nn_architectures.training_module import post_message_to_slack

# Set environment variables
ssm = boto3.client("ssm")

mine = ssm.get_parameter(Name='mine',
                         WithDecryption=True)["Parameter"]["Value"]
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

print(os.path.abspath(sys.argv[0]))

# S3
s3 = boto3.client("s3")
# leer data de las snapshots
filename = "data_snapshots.pkl"
path_data = "sagemaker_cosmos/" + filename
data = read_pkl_s3(os.environ["buckettrain"], path_data)
data.drop(columns=['prediccion'], inplace=True)
# Leer data del ads
filename = "ads_s3.pkl"
path_data = "sagemaker_cosmos/" + filename
ads = read_pkl_s3(os.environ["buckettrain"], path_data)
ads.sort_values(by=["equipo", "date"], inplace=True)
# Sacar las cargas anteriores en dos ciclos
ads = sacar_cargas_anteriores(ads)
# Eliminar outliers de ciclos de combustible
ads = ads_s3_filtros(ads)
# Delta days
delta_days = 3
# Delta de tiempo en minutos
delta_t = 30
timezone = pytz.timezone("America/Santiago")
# Fecha final
fecha_final = datetime.now(timezone)
fecha_final = fecha_final - timedelta(days=delta_days)
# Fecha inicial
fecha_inicial = '04-11-2019 00:00:00'
fecha_inicial = datetime.strptime(fecha_inicial, "%d-%m-%Y %H:%M:%S")
fecha_inicial = timezone_fechas("America/Santiago", fecha_inicial)
# Filtro en la temporalidad búscada
filter_ = ((ads['date'] >= fecha_inicial) & (ads['date'] <= fecha_final))
ads = ads[filter_]
ads = ads.sort_values(by=['date']).reset_index(drop=True)
# Cálculo de la tasa de consumo
ads['tasa_consumo'] = ads['cantidad'] / ads['t_max_encendido_cargas']
ads = ads[ads['carga_normal_anterior'] == True].reset_index(drop=True)

# Acá comienza el etiquetador de snapshots
ratios = []
database = pd.DataFrame()
for i in range(len(ads)):
    print('Iteración número: ', i)
    equipo_i = ads['equipo'].iloc[i]
    fecha1 = ads['date'].iloc[i]
    fecha2 = ads['fecha_camion_anterior'].iloc[i]

    # Filtros en los snapshots
    filtros = ((data['equipo'] == equipo_i) & (data['fecha_now'] <= fecha1) &
               (data['fecha_now'] >= fecha2))
    snap = data[filtros].sort_values(by=['fecha_now']).reset_index(drop=True)
    a1 = len(snap)

    # Cálculo de la tasa de consumo con el ads final
    tasa_i = ads['cantidad'].iloc[i] / ads['t_max_encendido_cargas'].iloc[i]
    t_max_encendido = ads['t_max_encendido_cargas'].iloc[i]

    # Carga normal anterior del ciclo para ponerla a tiempo real
    carga_normal_anterior_i = ads['carga_normal_A2'].iloc[i]
    cantidad_anterior_i = ads['Cantidad_A2'].iloc[i]
    # Filtrar la snapshot
    snap = snap[snap['t_max_encendido_cargas'] <= t_max_encendido]
    snap.rename(columns={'carga_normal':
                         'carga_normal_anterior',
                         'cantidad': 'cantidad_anterior'}, inplace=True)
    snap.reset_index(drop=True, inplace=True)
    a2 = len(snap)
    try:
        ratio = a2 / a1 * 100
    except Exception as e:
        print(e)
        if (a2 == 0) & (a1 != 0):
            ratio = 0
        elif (a1 == 0) & (a2 == 0):
            ratio = np.nan
        else:
            ratio = np.nan
    resta = a1 - a2
    # print("ratio: ", ratio)
    ratios.append([ratio, resta])

    if resta <= 2:
        # Cálculo de la tasa de consumo
        snap['cantidad'] = tasa_i * snap['t_max_encendido_cargas']
        snap['carga_normal'] = 'True'
        database = pd.concat([database, snap], axis=0)

# Guardar base de datos
database.sort_values(by=['equipo', 'fecha_now'], inplace=True)
database = database[(database["date"] >= fecha_inicial) &
                    (database["date"] <= fecha_final)]
database.reset_index(drop=True, inplace=True)

# Llevar a s3
filename = "database.pkl"
path_data = "sagemaker_cosmos/" + filename
database.to_pickle(filename)
s3.upload_file(Filename=f"{filename}",
               Bucket=os.environ["buckettrain"],
               Key=path_data)

space = "=============================================="
space = space + space
msg = f"({mine}) El proceso de etiquetado snapshots acaba de finalizar" +\
    " enviando" + '\n' + f"{filename} a S3 en la ruta {path_data}"
texto = space + '\n' + msg + '\n' + space
post_message_to_slack(texto)

# Ver gráficos usables
ratios = pd.DataFrame(ratios, columns=['ratio', 'resta'])
ratio1 = ratios[ratios['resta'] < 1]
usables = len(ratio1) / len(ratios)
print(usables)
