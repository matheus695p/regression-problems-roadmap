import os
import sys
import boto3
import warnings
from src.nn_architectures.training_module import (analisis_colinealidades,
                                                  read_pkl_s3,
                                                  post_message_to_slack)
# from datetime import datetime
warnings.filterwarnings("ignore")

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
print(os.path.abspath(sys.argv[0]))


# S3
s3 = boto3.client("s3")

# Cargar los ads directo de S3
filename = "ads_s3.pkl"
path_data = "sagemaker_cosmos/" + filename
ads = read_pkl_s3(os.environ["buckettrain"], path_data)

# Refiltrado con la data
fecha_lim = "2019-11-01"
ads = ads[ads['Date'] > fecha_lim].reset_index(drop=True)

# Filtrar ciertas variables de acuerdo a ciertas condiciones
ads = ads.sort_values(by=['date']).reset_index(drop=True)
ads = ads[(ads["cantidad"] > 0) & (ads['cantidad'] < 4542)]
ads = ads[ads['tiempo_ciclo_mean'] >= 0]
ads = ads[(ads["suma_toneladas"] > 0) & (ads['suma_toneladas'] < 10000)]
ads = ads[ads["diff_hrs"] < 60]
ads = ads[(ads['t_apagado'] < ads['diff_hrs']) &
          (ads['tiempo'] < ads['diff_hrs'])]
ads = ads[(ads['t_cargado'] >= 0) & (ads['t_cargado'] <= 48)]

# Eliminación de cargas anómalas
ads['tasa_de_consumo'] = ads['cantidad'] / ads['diff_hrs']
tasa_anomala = 240
tasas = ads[['tasa_de_consumo']]

# Eliminación de ciclos de combustible con cargas anomalas
ads = ads[ads['tasa_de_consumo'] <= tasa_anomala]
tasas = ads[['tasa_de_consumo']]

# ads = pd.concat([ads1, ads3], axis=0).reset_index(drop=True)
clases = ads.Cantidad.value_counts(bins=20).reset_index(drop=False)

# features
variables = ads.columns.to_list()
X = ads[variables]

# # Reemplazamos carga normal por valores 0 y 1
X['carga_normal'] = X['carga_normal'].apply(str)
X["carga_normal"] = X["carga_normal"].replace(["True", "False"], [1, 0])

X['carga_normal_A'] = X['carga_normal_A'].apply(str)
X["carga_normal_A"] = X["carga_normal_A"].replace(["True", "False"], [1, 0])

# Selecciónde variables
X = X[['Cantidad', 'Cantidad_A', 'suma toneladas', 't_final',
       'distancia_gps', 'num_cargado', 'diff hrs', 'num_registros',
       'num_cargado_subiendo', 'diff_cota_sub', 'diff_cota',
       'tiempo', 't_apagado', 'carga_normal', 'carga_normal_A', 't_cargado',
       'numero_cargas', 'vel_mean', 'vel025', 'vel075', 'tiempo_ciclo_mean',
       'distancia_pendiente', 't_encendido', 'velocidad_subiendo_c',
       'velocidad_subiendo_v', 'velocidad_bajando_c', 'velocidad_bajando_v',
       'velocidad_plano_c', 'velocidad_plano_v', 'segundos_subiendo_c',
       'segundos_subiendo_v', 'segundos_bajando_c', 'segundos_bajando_v',
       'segundos_plano_c', 'segundos_plano_v', 'dist_subiendo_c',
       'dist_subiendo_v', 'dist_bajando_c', 'dist_bajando_v', 'dist_plano_c',
       'dist_plano_v', 'angulo', 'aceleracion_positiva', 'fuerza_motriz_sum',
       'fuerza_motriz_mean', 'fuerza_subiendo_c', 'fuerza_subiendo_v',
       'fuerza_bajando_c', 'fuerza_bajando_v', 'fuerza_plano_c',
       'fuerza_plano_v', 't_cargas_diff', 'presion_total', 'presion_media',
       'temp_neumaticos', 't_cargado_mems', 'velocidad_origin_destination',
       'distancia_origin_destination', 't_encendido_mems',
       't_max_encendido_cargas']]
X = X.dropna()

X = X.reset_index(drop=True)
# Hacer el plot con el histograma de las variables
df = X.copy()
target = 'cantidad'
# Histograma de la variable cantidad
x1 = X[['cantidad']]
# Visualización de colinealidades
analisis_colinealidades(X, target)

# Guardar los datos preprocesados
filename = 'preprocessed_ads.pkl'
path_data = "sagemaker_cosmos/" + filename
X.to_pickle(filename)
s3.upload_file(Filename=f"{filename}",
               Bucket=os.environ["buckettrain"],
               Key=path_data)

columns = X.columns
largo = len(X)
space = "=============================================="
space = space + space
msg = f"({mine}) El proprocesamiento del ADS historico finalizó" +\
    " enviando" + '\n' + f"{filename} a S3 en la ruta {path_data}" + '\n' +\
    " Las columnas fueron:" + "\n" + f"{columns}" + '\n' + \
    f"El ADS historico tiene {largo} filas"
texto = space + '\n' + msg + '\n' + space
post_message_to_slack(texto)
