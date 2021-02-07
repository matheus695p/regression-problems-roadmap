import os
import sys
import boto3
import pytz
import pandas as pd
from datetime import timedelta, datetime
from src.nn_architectures.training_module import (post_message_to_slack,
                                                  read_pkl_s3, load_ads)

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

print(os.path.abspath(sys.argv[0]))
# S3
s3 = boto3.client("s3")
# Delta days
delta_days = 3
# Delta de tiempo en minutos
delta_t = 30

# timezone
timezone = pytz.timezone("America/Santiago")
# Fecha final
fecha_final = datetime.now(timezone).replace(tzinfo=None)
fecha_final = fecha_final - timedelta(days=delta_days)
# Dias de ads
dias_ads = 91
# Fecha en string para buscar el ads
fecha = datetime.strftime(fecha_final, "%Y-%m-%d")

# Cargar datos de octubre-noviembre-diciembre
s3 = boto3.client("s3")
fecha = datetime.strptime(fecha, "%Y-%m-%d")

# Cargar los últimos 45 días del ads
ads_nuevo = load_ads(s3, fecha, dias_ads, "indicadores/ads")
ads_nuevo.drop_duplicates(subset=["date", "equipo"], inplace=True)
ads_nuevo = ads_nuevo.sort_values(by=['date']).reset_index()
ads_nuevo.drop(columns=['index'], inplace=True)

# Leer el ads viejo
filename = "ads_s3.pkl"
path_data = "sagemaker_cosmos/" + filename
ads_viejo = read_pkl_s3(os.environ["buckettrain"], path_data)

# Concatenar con los nuevos ads
ads = pd.concat([ads_nuevo, ads_viejo], axis=0)
ads.drop_duplicates(subset=["equipo", "date"], inplace=True)

ads.to_pickle(filename)
s3.upload_file(Filename=f"{filename}",
               Bucket=os.environ["buckettrain"],
               Key=path_data)

space = "=============================================="
space = space + space
msg = f"({mine}) La concatenacion de ADS historicos acaba de finalizar" +\
    " enviando" + '\n' + f"{filename} a S3 en la ruta {path_data}"
texto = space + '\n' + msg + '\n' + space
post_message_to_slack(texto)
