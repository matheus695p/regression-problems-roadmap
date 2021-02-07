import pandas as pd
import io
import pytz
import boto3
import os
import pickle
from datetime import timedelta, datetime

# Set environment variables
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


def get_ads_status(s3, fecha_actual, dias, ruta_s3):
    """
    La funcion busca los archivos .parquet que cumplen un rango de
    fechas
    Parameters
    ----------
    s3: Conexion a servicio s3
    fecha_actual : Ultima fecha desde la que se analiza
    dias : Dias hacia atras que se quiere buscar
    ruta_s3 : carpeta s3 a la que se quiere ir
    Returns
    -------
    data_final : dataframe con los datos
    """
    # Data final
    data_final = pd.DataFrame()
    # Lista de fechas en el rango
    fechas = []
    bucket = os.environ["buckettrain"]
    # Calculamos los dias anteriores
    for i in range(dias):
        print(i)
        d = (fecha_actual-timedelta(i)).day
        m = (fecha_actual-timedelta(i)).month
        y = (fecha_actual-timedelta(i)).year
        prefijo = str(y) + "/" + str(m) + "/" + str(d) + "/"
        print(prefijo)
        fechas.append(prefijo)
    # Para cada dia
    for fecha in fechas:
        try:
            buffer = io.BytesIO()
            client = boto3.resource("s3")
            root = f"{ruta_s3}/{fecha}ads_status.parquet"
            print(root)
            object = client.\
                Object(bucket, root)
            object.download_fileobj(buffer)

            data = pd.read_parquet(buffer)
            data_final = pd.concat([data_final, data])

        except Exception as e:
            print(e)

            pass
    return data_final


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
    # Lista de fechas en el rango
    fechas = []
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
            path = f"{ruta_s3}/{prefijo}ads_modelo_no_predict_{d}-{m}-{y}.pkl"
            data = read_pkl_s3(bucket, path)
            print(path)
            data_final = pd.concat([data_final, data])
        except Exception as e:
            print(e)
            pass

    return data_final


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


def read_ads_snapshot_v2(fecha_inicial='01-02-2020 00:00:00',
                         fecha_final='31-07-2020 00:00:00',
                         delta_t=60):
    """
    Ir a leear la serie de snapshots de ads que se cargaron

    Parameters
    ----------
    fecha_inicial : TYPE, optional
        DESCRIPTION. The default is '01-02-2020 00:00:00'.
    fecha_final : TYPE, optional
        DESCRIPTION. The default is '31-07-2020 00:00:00'.

    Returns
    -------
    None.

    """
    fecha_inicial = datetime.strptime(fecha_inicial, "%d-%m-%Y %H:%M:%S")
    fecha_final = datetime.strptime(fecha_final, "%d-%m-%Y %H:%M:%S")
    salida = pd.DataFrame()
    while fecha_inicial <= fecha_final:
        fecha = fecha_inicial
        day = fecha.day
        month = fecha.month
        year = fecha.year
        hour = fecha.hour
        minute = fecha.minute

        name = f'{day}_{month}_{year}_{hour}_{minute}.pkl'
        root = f'ads_time_window_v2/{year}/{month}/{day}/{name}'
        print(root)
        try:
            data = read_pkl_s3(os.environ['outputbucket'], root)
            salida = pd.concat([salida, data], axis=0)
        except:
            print("===> el archivo no existe", root)
        fecha_inicial = fecha_inicial + timedelta(minutes=delta_t)
    return salida


def read_ads_snapshot_v3(fecha_inicial='01-02-2020 00:00:00',
                         fecha_final='31-07-2020 00:00:00',
                         delta_t=60):
    """
    Ir a leear la serie de snapshots de ads que se cargaron

    Parameters
    ----------
    fecha_inicial : TYPE, optional
        DESCRIPTION. The default is '01-02-2020 00:00:00'.
    fecha_final : TYPE, optional
        DESCRIPTION. The default is '31-07-2020 00:00:00'.

    Returns
    -------
    None.

    """
    fecha_inicial = datetime.strptime(fecha_inicial, "%d-%m-%Y %H:%M:%S")
    fecha_final = datetime.strptime(fecha_final, "%d-%m-%Y %H:%M:%S")
    salida = pd.DataFrame()
    while fecha_inicial <= fecha_final:
        fecha = fecha_inicial
        day = fecha.day
        month = fecha.month
        year = fecha.year
        hour = fecha.hour
        minute = fecha.minute

        name = f'{day}_{month}_{year}_{hour}_{minute}.pkl'
        root = f'ads_time_window_v2/{year}/{month}/{day}/{name}'
        print(root)
        try:
            data = read_pkl_s3(os.environ['outputbucket'], root)
            salida = pd.concat([salida, data], axis=0)
        except:
            print("===> el archivo no existe", root)
        fecha_inicial = fecha_inicial + timedelta(minutes=delta_t)
    return salida


def ads_s3_filtros(ads):
    """
    Filtros aplicados al ads para entrenar sin datos anomalos

    Parameters
    ----------
    ads : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ads['tasa_de_consumo'] = ads['cantidad'] / ads['diff_hrs']
    tasa_anomala = 240
    # Eliminación de ciclos de combustible con cargas anomalas
    ads = ads[ads['tasa_de_consumo'] <= tasa_anomala]
    ads['mayor_1500'] = ads['cantidad'] > 1500
    # Filtro por diferencia de horas
    ads1 = ads[(ads['mayor_1500'] == False) & (ads['diff_hrs'] < 11)]
    ads2 = ads[(ads['mayor_1500'] == True) & (ads['diff_hrs'] >= 11)]
    ads = pd.concat([ads1, ads2], axis=0).reset_index(drop=True)
    # Filtro por tonelajes
    ads1 = ads[(ads['mayor_1500'] == False) &
               (ads['suma_toneladas'] < 4000)]
    ads2 = ads[ads['mayor_1500'] == True]
    ads = pd.concat([ads1, ads2], axis=0).reset_index(drop=True)
    # FIltros de seguridad
    ads = ads[(ads["cantidad"] > 50) & (ads['cantidad'] < 4542)]
    ads = ads[ads['t_cargado'] >= 0]
    ads = ads[ads['tiempo_ciclo_mean'] >= 0]
    ads = ads[(ads["suma_toneladas"] > 300) & (ads['suma_toneladas'] < 8000)]
    ads = ads[ads["diff_hrs"] < 60]
    ads.drop(columns=['tasa_de_consumo', 'mayor_1500'], inplace=True)
    return ads


def timezone_fechas(zona, fecha):
    """
    La funcion entrega el formato de zona horaria a las fechas de los
    dataframe
    Parameters
    ----------
    zona: zona horaria a usar
    fecha: fecha a modificar
    Returns
    -------
    fecha_zh: fecha con el fomarto de zona horaria
    """
    # Definimos la zona horaria
    timezone = pytz.timezone(zona)
    fecha_zs = timezone.localize(fecha)

    return fecha_zs


def arreglar_t(df):
    """
    El máximo entre t_encendido y t_diff_cargas debe ser un número creciente

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for i in range(len(df)):
        if i > 0:
            df['t_max_encendido_cargas'].iloc[i] =\
                max(df['t_max_encendido_cargas'].iloc[i],
                    df['t_max_encendido_cargas'].iloc[i-1])
    return df


def sacar_cargas_anteriores(ads):
    """
    Sacar cargas anteriores

    Parameters
    ----------
    ads : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    for variable in ads.columns:
        if variable in ["Cantidad_A", "carga_normal_A"]:
            new_parametro = variable.replace('_A', '') + "_A2"
            ads[new_parametro] = ads.groupby(by=["Equipo"]).shift()[variable]
    return ads
