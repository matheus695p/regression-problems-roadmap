import boto3
from module_main import (train_nn_cosmos, verificar_a_en_b,
                         obtener_nombre_modelo, reordenar_fuentes,
                         columnas_fuentes, modelo_gps_loads_mems_specto,
                         modelo_gps_loads_specto, modelo_gps_mems_specto,
                         modelo_loads_mems_specto, modelo_gps_loads_mems,
                         modelo_gps_loads, modelo_gps_mems, modelo_loads_mems,
                         modelo_loads_specto, modelo_mems_specto,
                         modelo_gps_specto, modelo_gps,
                         modelo_loads, modelo_mems, modelo_specto)


ssm = boto3.client("ssm")
fuentes_dispo = ssm.get_parameter(Name='fuentes_disponibles',
                                  WithDecryption=True)["Parameter"]["Value"]
lista_fuentes = fuentes_dispo.split(",")
# reordenar fuentes
lista_fuentes = reordenar_fuentes(lista_fuentes)
combinaciones = [['gps', 'loads', 'mems', 'specto'],
                 ['gps', 'loads', 'specto'],
                 ['gps', 'mems', 'specto'],
                 ['loads', 'mems', 'specto'],
                 ['gps', 'loads', 'mems'],
                 ['gps', 'loads'],
                 ['gps', 'mems'],
                 ['gps', 'specto'],
                 ['loads', 'mems'],
                 ['loads', 'specto'],
                 ['mems', 'specto'],
                 ['gps'],
                 ['loads'],
                 ['mems'],
                 ['specto']]

# obtener nombre de todos los modelos de acuerdo a las fuentes
nombre_modelos = []
# diccionario con nombre del modelo y las fuentes
nombre_modelo_fuente = {}
for combinacion in combinaciones:
    if verificar_a_en_b(combinacion, lista_fuentes):
        nombre = obtener_nombre_modelo(combinacion)
        nombre_modelos.append(nombre)
        nombre_modelo_fuente[nombre] = combinacion
# comienza entrenamiento modelos
# se podria hacer con un loop pero en el caso de modificar un modelo
# independiente
if 'modelo_gps_loads_mems_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_loads_mems_specto'])
    n_features = len(columns) - 1
    model = modelo_gps_loads_mems_specto(n_features)
    train_nn_cosmos('modelo_gps_loads_mems_specto', columns, model)
if 'modelo_gps_loads_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_loads_specto'])
    n_features = len(columns) - 1
    model = modelo_gps_loads_specto(n_features)
    train_nn_cosmos('modelo_gps_loads_specto', columns, model)
if 'modelo_gps_mems_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_mems_specto'])
    n_features = len(columns) - 1
    model = modelo_gps_mems_specto(n_features)
    train_nn_cosmos('modelo_gps_mems_specto', columns, model)
if 'modelo_loads_mems_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_loads_mems_specto'])
    n_features = len(columns) - 1
    model = modelo_loads_mems_specto(n_features)
    train_nn_cosmos('modelo_loads_mems_specto', columns, model)
if 'modelo_gps_loads_mems' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_loads_mems'])
    n_features = len(columns) - 1
    model = modelo_gps_loads_mems(n_features)
    train_nn_cosmos('modelo_gps_loads_mems', columns, model)
if 'modelo_gps_loads' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_loads'])
    n_features = len(columns) - 1
    model = modelo_gps_loads(n_features)
    train_nn_cosmos('modelo_gps_loads', columns, model)
if 'modelo_gps_mems' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_mems'])
    n_features = len(columns) - 1
    model = modelo_gps_mems(n_features)
    train_nn_cosmos('modelo_gps_mems', columns, model)
if 'modelo_loads_mems' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_loads_mems'])
    n_features = len(columns) - 1
    model = modelo_loads_mems(n_features)
    train_nn_cosmos('modelo_loads_mems', columns, model)
if 'modelo_loads_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_loads_specto'])
    n_features = len(columns) - 1
    model = modelo_loads_specto(n_features)
    train_nn_cosmos('modelo_loads_specto', columns, model)
if 'modelo_mems_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_mems_specto'])
    n_features = len(columns) - 1
    model = modelo_mems_specto(n_features)
    train_nn_cosmos('modelo_mems_specto', columns, model)
if 'modelo_gps_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps_specto'])
    n_features = len(columns) - 1
    model = modelo_gps_specto(n_features)
    train_nn_cosmos('modelo_gps_specto', columns, model)
if 'modelo_gps' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_gps'])
    n_features = len(columns) - 1
    model = modelo_gps(n_features)
    train_nn_cosmos('modelo_gps', columns, model)
if 'modelo_loads' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_loads'] - 1)
    n_features = len(columns) - 1
    model = modelo_loads(n_features)
    train_nn_cosmos('modelo_loads', columns, model)
if 'modelo_mems' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_mems'])
    n_features = len(columns) - 1
    model = modelo_mems(n_features)
    train_nn_cosmos('modelo_mems', columns, model)
if 'modelo_specto' in nombre_modelos:
    columns = columnas_fuentes(
        nombre_modelo_fuente['modelo_specto'])
    n_features = len(columns) - 1
    model = modelo_specto(n_features)
    train_nn_cosmos('modelo_specto', columns, model)
