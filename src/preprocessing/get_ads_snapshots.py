import os
import sys
import pytz
import boto3
import pandas as pd
from datetime import datetime, timedelta
from src.preprocessing.labeller_module import (read_ads_snapshot_v2)
from src.nn_architectures.training_module import (post_message_to_slack,
                                                  read_pkl_s3)


ssm = boto3.client("ssm")
# bucket de entrenamiento
os.environ["buckettrain"] =\
    ssm.get_parameter(Name='TrainBucket-develop',
                      WithDecryption=True)["Parameter"]["Value"]
# nombre de la mina
mine = ssm.get_parameter(Name='mine',
                         WithDecryption=True)["Parameter"]["Value"]

print(os.path.abspath(sys.argv[0]))


space = "=============================================="
space = space + space
msg = f"({mine}) Un nuevo proceso de reentrenamiento acaba de comenzar"
texto = space + '\n' + msg + '\n' + space
post_message_to_slack(texto)

# Llamar al cliente
s3 = boto3.client("s3")
# Delta days
delta_days = 3
# Delta de tiempo en MINUTOS
delta_t = 30
# Timezone
timezone = pytz.timezone("America/Santiago")
# Fecha final
fecha_final = datetime.now(timezone).replace(tzinfo=None)
fecha_final = fecha_final - timedelta(days=delta_days)
# Reemplazar el horario por las 12 de la noche
fecha_final = fecha_final.replace(hour=0, minute=0, second=0)
# Fecha inicial es un mes hacia atrás nomás, no entrenar superior a eso
dias_ads = 90
fecha_inicial = fecha_final - timedelta(days=dias_ads)
# Fecha inicial
fecha_final = fecha_final.strftime("%d-%m-%Y %H:%M:%S")
fecha_inicial = fecha_inicial.strftime("%d-%m-%Y %H:%M:%S")
# Leer ads del snapshot
data_nueva = read_ads_snapshot_v2(fecha_inicial=fecha_inicial,
                                  fecha_final=fecha_final,
                                  delta_t=delta_t)
data_nueva.reset_index(drop=True, inplace=True)
# Leer la data actual de snapshots
filename = "data_snapshots.pkl"
path_data = "sagemaker_cosmos/" + filename
data_actual = read_pkl_s3(os.environ["buckettrain"], path_data)
# concatenación de la data actual con la antigua
data = pd.concat([data_nueva, data_actual], axis=0)
data.drop_duplicates(subset=["equipo", "fecha_now"], inplace=True)
data.reset_index(drop=True, inplace=True)
# Llevar a S3
data.to_pickle(filename)
s3.upload_file(Filename=f"{filename}",
               Bucket=os.environ["buckettrain"],
               Key=path_data)

space = "=============================================="
space = space + space
msg = f"({mine}) La concatenacion de ADS snapshots acaba de finalizar" +\
    " enviando" + '\n' + f"{filename} a S3 con ruta {path_data}"
texto = space + '\n' + msg + '\n' + space
post_message_to_slack(texto)
