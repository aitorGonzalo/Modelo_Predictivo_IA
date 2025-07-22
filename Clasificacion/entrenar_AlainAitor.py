# Imports del script
import os
import shutil
from sys import exit, version_info
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import sys
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



# Variables globales
OUTPUT_FILE     = "./output"                    # Path del archivo de salida
INPUT_FILE      = "iris.csv"                    # Path del archivo de entrada
TARGET_NAME     = "Especie"                      # Nombre de la columna a clasificar
K_MIN           = 1                             # Numero minimo de "nearest neighbors"
K_MAX           = 5                             # Número máximo de "nearest neighbors"
K_AUMENTO       = 2
D               = ['uniform', 'distance']       # Ponderacion de distancias
P_MIN           = 2                             # Tipo de distancia -> 1: Manhatan | 2: Euclídea
P_MAX           = 2                             # Tipo de distancia -> 1: Manhatan | 2: Euclídea
P_AUMENTO       = 1

DEV_SIZE        = 0.2                           # Indice del tamaño del dev. Por defecto un 20% de la muestra
RANDOM_STATE    = 42                            # Seed del random split
ALGORITMO_A_USAR = "KNN"
BONANZA = "f1-score"
METODO_REESCALADO = "MINMAX"
HACER_PREPROCESADO = True
METODO_PREPROCESADO = "Impute"
IMPUTAR_CON ="MEAN"
MIN_DEPTH = 1
MAX_DEPTH = 3
DEPTH_AUMENTO = 1
MIN_SAMPLE_LEAF = 1
MAX_SAMPLE_LEAF = 3
SAMPLE_LEAF_AUMENTO = 1
CRITERION = "gini"
MIN_N_ESTIMATOR = 1
MAX_N_ESTIMATOR = 3
N_ESTIMATOR_AUMENTO = 1
CUANTOS_ATRIBUTOS_SELECCIONAS = "todos"
CUALES_DESCARTAS_O_ELIGES = ["Largo de sepalo"]
UNDERSAMPLING_OVERSAMPLING = "undersampling"
RESULTADOS_MODELO = []
AVG = "macro"
CONSTANT = 1

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: entrenar_knn.py <optional-args>")
    print("In order to use this pyhton program, you need to feel config.json with the following settings:")
    print(f"INPUT_FILE     input file path of the data                 DEFAULT: ./{INPUT_FILE}")
    print(f"OUTPUT_FILE    output file path for the weights            DEFAULT: ./{OUTPUT_FILE}")
    print(f"TARGET_NAME    name of the target variable                DEFAULT: ./{TARGET_NAME}")
    print(f"D             distance parameter ->                       DEFAULT: {D}")
    print(f"K_MIN         number of neighbors for the KNN algorithm   DEFAULT: {K_MIN}")
    print(f"K_MAX        number of neighbors for the KNN algorithm   DEFAULT: {K_MAX}")
    print(f"K_AUMENTO        increment of the variable K_MIN and K_MAX   DEFAULT: {K_AUMENTO}")
    print(f"P_MIN         distance from -> 1: Manhatan | 2: Euclidean DEFAULT: {P_MIN}")
    print(f"P_MAX          distance to -> 1: Manhatan | 2: Euclidean   DEFAULT: {P_MIN}")
    print(f"P_AUMENTO         increment of the variable P_MIN and P_MAX DEFAULT: {P_AUMENTO}")
    print(f"DEV_SIZE        index of size of the dev          DEFAULT: {DEV_SIZE}")
    print(f"RANDOM_STATE   random seed of the split         DEFAULT: {RANDOM_STATE}")
    print(f"ALGORITMO_A_USAR   algorithm used          DEFAULT: {ALGORITMO_A_USAR}")
    print(f"BONANZA     quality of the model         DEFAULT: {BONANZA}")
    print(f"METODO_REESCALADO     rescale method used        DEFAULT: {METODO_REESCALADO}")
    print(f"HACER_PREPROCESADO     preprocess the data ---> yes|true / no|false        DEFAULT: {HACER_PREPROCESADO}")
    print(f"METODO_PREPROCESADO    shows the method selected ---> Impute/Drop         DEFAULT: {METODO_PREPROCESADO}")
    print(f"IMPUTAR_CON            the method you wanna impute the missing values ---> MEAN/MEDIAN/MODE/CONSTANT     DEFAULT: {IMPUTAR_CON}")
    print(f"CONSTANT     value to imput with cont¡stant method      DEFAULT: {AVG}")
    print(f"MIN_DEPTH     minimum depth of the trees         DEFAULT: {MIN_DEPTH}")
    print(f"MAX_DEPTH     maximum depth of the trees         DEFAULT: {MAX_DEPTH}")
    print(f"DEPTH_AUMENTO     increment step for depth in the parameter sweep         DEFAULT: {DEPTH_AUMENTO}")
    print(f"MIN_SAMPLE_LEAF     minimum samples per leaf         DEFAULT: {MIN_SAMPLE_LEAF}")
    print(f"MAX_SAMPLE_LEAF     maximum samples per leaf         DEFAULT: {MAX_SAMPLE_LEAF}")
    print(f"SAMPLE_LEAF_AUMENTO     increment step for sample per leaf in the parameter sweep         DEFAULT: {SAMPLE_LEAF_AUMENTO}")
    print(f"CRITERION     criterion for splitting -->gini/entrophy        DEFAULT: {CRITERION}")
    print(f"MIN_N_ESTIMATOR     minimum number of estimators         DEFAULT: {MIN_N_ESTIMATOR}")
    print(f"MAX_N_ESTIMATOR     maximum number of estimators         DEFAULT: {MAX_N_ESTIMATOR}")
    print(f"N_ESTIMATOR_AUMENTO     increment step for number of estimators in the parameter sweep         DEFAULT: {N_ESTIMATOR_AUMENTO}")
    print(f"CUANTOS_ATRIBUTOS_SELECCIONAS     how many attributes are selected for the model        DEFAULT: {CUANTOS_ATRIBUTOS_SELECCIONAS}")
    print(f"CUALES_DESCARTAS_O_ELIGES     which attributes are discarded or chosen explicitly        DEFAULT: {CUALES_DESCARTAS_O_ELIGES}")
    print(f"UNDERSAMPLING_OVERSAMPLING     whether to perform undersampling or oversampling         DEFAULT: {UNDERSAMPLING_OVERSAMPLING}")
    print(f"RESULTADOS_MODELO     where to store the model's results        DEFAULT: {RESULTADOS_MODELO}")
    print(f"AVG     the averaging method for performance metrics ---> macro,weighted      DEFAULT: {AVG}")



    exit(1)

def load_options():
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_FILE, OUTPUT_FILE, K_MIN, K_MAX, K_AUMENTO, D, P_MIN, P_MAX, P_AUMENTO, DEV_SIZE, RANDOM_STATE, ALGORITMO_A_USAR, CRITERION, AVG
    global BONANZA, IMPUTAR_CON, TARGET_NAME, METODO_REESCALADO, HACER_PREPROCESADO, METODO_PREPROCESADO, CONSTANT
    global MIN_DEPTH, MAX_DEPTH, DEPTH_AUMENTO, MIN_SAMPLE_LEAF, MAX_SAMPLE_LEAF, SAMPLE_LEAF_AUMENTO, MIN_N_ESTIMATOR, MAX_N_ESTIMATOR, N_ESTIMATOR_AUMENTO, CUALES_DESCARTAS_O_ELIGES, CUANTOS_ATRIBUTOS_SELECCIONAS, UNDERSAMPLING_OVERSAMPLING

    if len(sys.argv) > 1:

        if '-h' in sys.argv or '--help' in sys.argv:
            usage()
        else:
            config_path = sys.argv[1]
            with open(config_path, 'r') as config_file:
                json_config = json.load(config_file)
            if 'D' in json_config and all(d in ['uniform', 'distance'] for d in json_config['D']):
                D = json_config["D"]
            else:
                print("Valor no válido para D; usando el valor predeterminado.")

            INPUT_FILE = json_config['INPUT_FILE']


            if 'K_MIN' in json_config and isinstance(json_config['K_MIN'], int) and json_config['K_MIN'] > 0:
                K_MIN = json_config['K_MIN']
            else:
                print("Valor no válido para K_MIN; usando el valor predeterminado.")

            if 'K_MAX' in json_config and isinstance(json_config['K_MAX'], int) and json_config['K_MAX'] >= K_MIN:
                K_MAX = json_config['K_MAX']
            else:
                print("Valor no válido para K_MAX; usando el valor predeterminado.")

            if 'K_AUMENTO' in json_config and isinstance(json_config['K_AUMENTO'], int) and json_config['K_MAX'] >= K_MAX - K_MIN:
                K_AUMENTO = json_config["K_AUMENTO"]
            else:
                print("Valor no válido para K_AUMENTO; usando el valor predeterminado.")

            OUTPUT_FILE = json_config["OUTPUT_FILE"]

            if 'P_MIN' in json_config and isinstance(json_config['P_MIN'], int) and json_config['P_MIN'] > 0:
                P_MIN = json_config['P_MIN']
            else:
                print("Valor no válido para P_MIN; usando el valor predeterminado.")

            if 'P_MAX' in json_config and isinstance(json_config['P_MAX'], int) and json_config['P_MAX'] >= P_MIN:
                P_MAX = json_config['P_MAX']
            else:
                print("Valor no válido para P_MAX; usando el valor predeterminado.")

            if 'P_AUMENTO' in json_config and isinstance(json_config['P_AUMENTO'], int) and json_config['P_MAX'] >= P_MAX - P_MIN:
                P_AUMENTO = json_config["P_AUMENTO"]
            else:
                print("Valor no válido para P_AUMENTO; usando el valor predeterminado.")

            if 'DEV_SIZE' in json_config and isinstance(json_config['DEV_SIZE'], float):
                DEV_SIZE = json_config['DEV_SIZE']
            else:
                print("Valor no válido para DEV_SIZE; usando el valor predeterminado.")

            if 'RANDOM_STATE' in json_config and isinstance(json_config['RANDOM_STATE'], int):
                RANDOM_STATE = json_config['RANDOM_STATE']
            else:
                print("Valor no válido para RANDOM_STATE; usando el valor predeterminado.")

            if 'ALGORITMO_A_USAR' in json_config and json_config['ALGORITMO_A_USAR'] in ['KNN', 'decisiontree','randomforest']:
                ALGORITMO_A_USAR = json_config['ALGORITMO_A_USAR']
            else:
                print("Valor no válido para ALGORITMO_A_USAR; usando el valor predeterminado.")

            if 'BONANZA' in json_config and json_config['BONANZA'] in ["f1-score", "precision", "recall"]:
                BONANZA = json_config['BONANZA']
            else:
                print("Valor no válido para BONANZA; usando el valor predeterminado.")

            if 'TARGET_NAME' in json_config and isinstance(json_config['TARGET_NAME'], str):
                TARGET_NAME = json_config['TARGET_NAME']
            else:
                print("Valor no válido para TARGET_NAME; usando el valor predeterminado.")

            if 'METODO_PREPROCESADO' in json_config and json_config['METODO_PREPROCESADO'] in ['Impute','Drop']:
                METODO_PREPROCESADO = json_config["METODO_PREPROCESADO"]
            else:
                print("Valor no válido para METODO_PREPROCESADO; usando el valor predeterminado.")

            if 'HACER_PREPROCESADO' in json_config and isinstance(json_config['HACER_PREPROCESADO'], bool):
                HACER_PREPROCESADO = json_config['HACER_PREPROCESADO']
            else:
                print("Valor no válido para HACER_PREPROCESADO; usando el valor predeterminado.")

            if 'METODO_REESCALADO' in json_config and isinstance(json_config['METODO_REESCALADO'], str):
                METODO_REESCALADO = json_config['METODO_REESCALADO']
            else:
                print("Valor no válido para METODO_REESCALADO; usando el valor predeterminado.")

            if 'IMPUTAR_CON' in json_config and json_config['IMPUTAR_CON'] in ['MEAN','MEDIAN','MODE','CONSTANT']:
                IMPUTAR_CON = json_config["IMPUTAR_CON"]
            else:
                print("Valor no válido para IMPUTAR_CON; usando el valor predeterminado.")

            if 'CONSTANT' in json_config and isinstance(json_config['CONSTANT'], int):
                CONSTANT = json_config['CONSTANT']
            else:
                print("Valor no válido para RANDOM_STATE; usando el valor predeterminado.")

            if 'MIN_DEPTH' in json_config and isinstance(json_config['MIN_DEPTH'], int) and json_config['MIN_DEPTH'] > 0:
                MIN_DEPTH = json_config['MIN_DEPTH']
            else:
                print("Valor no válido para MIN_DEPTH; usando el valor predeterminado.")

            if 'MAX_DEPTH' in json_config and isinstance(json_config['MAX_DEPTH'], int) and json_config['MAX_DEPTH'] >= MIN_DEPTH:
                MAX_DEPTH = json_config['MAX_DEPTH']
            else:
                print("Valor no válido para MAX_DEPTH; usando el valor predeterminado.")

            if 'DEPTH_AUMENTO' in json_config and isinstance(json_config['DEPTH_AUMENTO'], int) and json_config['DEPTH_AUMENTO'] <= MAX_DEPTH - MIN_DEPTH:
                DEPTH_AUMENTO = json_config["DEPTH_AUMENTO"]
            else:
                print("Valor no válido para DEPTH_AUMENTO; usando el valor predeterminado.")

            if 'MIN_SAMPLE_LEAF' in json_config and isinstance(json_config['MIN_SAMPLE_LEAF'], int) and json_config['MIN_SAMPLE_LEAF'] > 0:
                MIN_SAMPLE_LEAF = json_config['MIN_SAMPLE_LEAF']
            else:
                print("Valor no válido para MIN_SAMPLE_LEAF; usando el valor predeterminado.")

            if 'MAX_SAMPLE_LEAF' in json_config and isinstance(json_config['MAX_SAMPLE_LEAF'], int) and json_config['MAX_SAMPLE_LEAF'] >= MIN_SAMPLE_LEAF:
                MAX_SAMPLE_LEAF = json_config['MAX_SAMPLE_LEAF']
            else:
                print("Valor no válido para MAX_SAMPLE_LEAF; usando el valor predeterminado.")

            if 'SAMPLE_LEAF_AUMENTO' in json_config and isinstance(json_config['SAMPLE_LEAF_AUMENTO'], int) and json_config['SAMPLE_LEAF_AUMENTO'] <= MAX_SAMPLE_LEAF - MIN_SAMPLE_LEAF:
                SAMPLE_LEAF_AUMENTO = json_config["SAMPLE_LEAF_AUMENTO"]
            else:
                print("Valor no válido para SAMPLE_LEAF_AUMENTO; usando el valor predeterminado.")

            if 'CRITERION' in json_config and json_config['CRITERION'] in ['gini', 'entropy']:
                CRITERION = json_config["CRITERION"]
            else:
                print("Valor no válido para CRITERION; usando el valor predeterminado.")

            if 'MIN_N_ESTIMATOR' in json_config and isinstance(json_config['MIN_N_ESTIMATOR'], int) and json_config['MIN_N_ESTIMATOR'] > 0:
                MIN_N_ESTIMATOR = json_config['MIN_N_ESTIMATOR']
            else:
                print("Valor no válido para MIN_N_ESTIMATOR; usando el valor predeterminado.")

            if 'MAX_N_ESTIMATOR' in json_config and isinstance(json_config['MAX_N_ESTIMATOR'], int) and json_config['MAX_N_ESTIMATOR'] >= MIN_N_ESTIMATOR:
                MAX_N_ESTIMATOR = json_config['MAX_N_ESTIMATOR']
            else:
                print("Valor no válido para MAX_N_ESTIMATOR; usando el valor predeterminado.")
            N_ESTIMATOR_AUMENTO = json_config["N_ESTIMATOR_AUMENTO"]
            if 'N_ESTIMATOR_AUMENTO' in json_config and isinstance(json_config['N_ESTIMATOR_AUMENTO'], int) and json_config['N_ESTIMATOR_AUMENTO'] <= MAX_N_ESTIMATOR - MIN_N_ESTIMATOR:
                N_ESTIMATOR_AUMENTO = json_config["N_ESTIMATOR_AUMENTO"]
            else:
                print("Valor no válido para N_ESTIMATOR_AUMENTO; usando el valor predeterminado.")
            CUANTOS_ATRIBUTOS_SELECCIONAS = json_config["CUANTOS_ATRIBUTOS_SELECCIONAS"]
            CUALES_DESCARTAS_O_ELIGES = json_config["CUALES_DESCARTAS_O_ELIGES"]

            if 'UNDERSAMPLING_OVERSAMPLING' in json_config and json_config['UNDERSAMPLING_OVERSAMPLING'] in ['undersampling','oversampling']:
                UNDERSAMPLING_OVERSAMPLING = json_config["UNDERSAMPLING_OVERSAMPLING"]
            else:
                print("Valor no válido para UNDERSAMPLING_OVERSAMPLING; usando el valor predeterminado.")

            if 'AVG' in json_config and json_config['AVG'] in ['macro','weighted']:
                AVG = json_config["AVG"]
            else:
                print("Valor no válido para AVG; usando el valor predeterminado.")
    else:
        exit()
def show_script_options():
    # PRE: ---
    # POST: imprimimos la configuración del script
    print("entrenar_knn.py configuration:")
    print(f"-d distance parameter   -> {D}")
    print(f"-i input file path      -> {INPUT_FILE}")
    print(f"-k number of neighbors  -> from: {K_MIN} to: {K_MAX}")
    print(f"-o output file path     -> {OUTPUT_FILE}")
    print(f"-p distance algorithm   -> from: {P_MIN} to: {P_MAX}")
    print(f"-alg algorithm to use   -> {ALGORITMO_A_USAR}")
    print(f"-bon quality of the model   -> {BONANZA}")
    print(f"-resc rescale method   -> {METODO_REESCALADO}")
    print(f"-prep preprocess data   -> {HACER_PREPROCESADO}")
    print(f"-prep preprocessing method   -> {METODO_PREPROCESADO}")
    print(f"-imp with what to impute   -> {IMPUTAR_CON}")
    print(f"-depth minimum tree depth   -> from: {MIN_DEPTH} to: {MAX_DEPTH}")
    print(f"-samples per leaf   -> from: {MIN_SAMPLE_LEAF} to: {MAX_SAMPLE_LEAF}")
    print(f"-leafinc samples per leaf increment   -> {SAMPLE_LEAF_AUMENTO}")
    print(f"-crit splitting criterion   -> {CRITERION}")
    print(f"-number of estimators   -> from: {MIN_N_ESTIMATOR} to: {MAX_N_ESTIMATOR}")
    print(f"-estinc estimator increment   -> {N_ESTIMATOR_AUMENTO}")
    print(f"-selattr number of attributes to select   -> {CUANTOS_ATRIBUTOS_SELECCIONAS}")
    print(f"-discardorchoose attributes to discard or choose   -> {CUALES_DESCARTAS_O_ELIGES}")
    print(f"-underover undersampling or oversampling   -> {UNDERSAMPLING_OVERSAMPLING}")
    print(f"-avg averaging method for metrics   -> {AVG}")
    print("")

#######################################################################################
#                               METHODS AND FUNCTIONS                                 #
#######################################################################################
def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    
    # Si no es anterior a la version 3 de python
    return str(x)

def atributos_excepto(atributos, excepciones):
    # PRE: lista completa de atributos y lista de aquellos que no queremos seleccionar
    # POST: devolvemos una lista de atributos
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)

    return atribs

def imprimir_atributos(atributos):
    # PRE: lista de atributos
    # POST: se imprime por pantalla la lista
    string = ""
    for atr in atributos:
        string += str(f"{atr} ")
    print("---- Atributos seleccionados")
    print(string)
    print()
def datetime_to_epoch(dt):
    # PRE: dt es un objeto de fecha o una serie de fechas en formato datetime64[ns].
    # POST: La función devuelve el valor de época (epoch) correspondiente a la fecha dada en dt.

    epoch = pd.Timestamp(dt)
    return epoch.timestamp()
def estandarizar_tipos_de_datos(dataset, categorical_features, numerical_features, text_features):
    # PRE: dataset y listas qué atributos son categóricos, numéricos y de texto del dataset
    # POST: devuelve las features categoriales y de texto en formato unicode y las numéricas en formato double o epoch (si son fechas)
    for feature in categorical_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(dataset[feature].dtype, 'base') and dataset[feature].dtype.base == np.dtype('M8[ns]')):
            dataset[feature] = datetime_to_epoch(dataset[feature])
        else:
            dataset[feature] = dataset[feature].astype('double')

def obtener_lista_impute_para(atributos, impute_with, excepciones):
    # PRE: lista de atributos y string indicando con qué valor los imputamos
    # POST: lista del estilo: [{"feature": atrib[i], "impute_with": impute_with}]
    lista = []
    for a in atributos:
        if a not in excepciones:
            entrada = {"feature" : a, "impute_with": impute_with}
            lista.append(entrada)

    return lista

def obtener_lista_rescalado_para(atributos, rescale_with, excepciones):
    # PRE: lista de atributos y string indicando con qué valor reescalamos
    # POST: diccionario del estilo: {'num_var45_ult1': 'AVGSTD', ... }

    diccionario = {}
    for a in atributos:
        if a not in excepciones:
            diccionario[a] = rescale_with

    return diccionario

def preprocesar_datos(dataset, drop_rows_when_missing, impute_when_missing, rescale_features):
    # PRE: Conjunto completo de datos para ajustar nuestro algoritmo
    # POST: Devuelve dos conjuntos: Train y Dev tratando los missing values y reescalados

    # Dividimos nuestros datos de entrenamiento en train y dev
    train, dev = train_test_split(dataset,test_size=DEV_SIZE,random_state=RANDOM_STATE,stratify=dataset[['__target__']])

    #Borrar valores faltantes
    if METODO_PREPROCESADO == "Drop":
          for feature in drop_rows_when_missing:
            train = train[train[feature].notnull()]
            dev = dev[dev[feature].notnull()]
            print('Dropped missing records in %s' % feature)

    #Imputar valores faltantes
    elif METODO_PREPROCESADO == "Impute":
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN': #Con la media
                v1 = train[feature['feature']].mean()
                v2 = dev[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN': #Con la mediana
                v1 = train[feature['feature']].median()
                v2 = dev[feature['feature']].median()
            elif feature['impute_with'] == 'MODE': #Con la moda
                v1 = train[feature['feature']].value_counts().index[0]
                v2 = dev[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v1 = CONSTANT
                v2 = v1
            train[feature['feature']] = train[feature['feature']].fillna(v1)
            dev[feature['feature']] = dev[feature['feature']].fillna(v2)

            s1 = f"- Train feature {feature['feature']} with value {str(v1)}"
            s2 = f"- Dev feature {feature['feature']} with value {str(v2)}"
            print("Imputed missing values\t%s\t%s" % (s1, s2))


    #Reescalamos los datos en funcion del parametro introducido
    for (feature_name, rescale_method) in rescale_features.items():
        # Obtenemos los valores de rescalado de dev y test
        if rescale_method == 'MINMAX':
            # Valores del train
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale1 = _max - _min
            shift1 = _min
            # Valores del dev
            _min = dev[feature_name].min()
            _max = dev[feature_name].max()
            scale2 = _max - _min
            shift2 = _min

        else:
            # Valores del train
            scale1 = train[feature_name].std()
            shift1 = train[feature_name].mean()
            # Valores del dev
            scale2 = dev[feature_name].std()
            shift2 = dev[feature_name].mean()
        
    # Rescalamos dev y test
    if scale1 == 0. or scale2 == 0.:
        del train[feature_name]
        del dev[feature_name]
        print('Feature %s was dropped because it has no variance in train or dev' % feature_name)
    else:
        print('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift1).astype(np.float64) / scale1
        dev[feature_name] = (dev[feature_name] - shift2).astype(np.float64) / scale2


    for (feature_name, rescale_method) in rescale_features.items():
        # Obtenemos los valores de rescalado de dev y test
        if rescale_method == 'MINMAX':
            # Valores del train
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale1 = _max - _min
            shift1 = _min
            # Valores del dev
            _min = dev[feature_name].min()
            _max = dev[feature_name].max()
            scale2 = _max - _min
            shift2 = _min

        else:
            # Valores del train
            scale1 = train[feature_name].std()
            shift1 = train[feature_name].mean()
            # Valores del dev
            scale2 = dev[feature_name].std()
            shift2 = dev[feature_name].mean()

    # Rescalamos dev y test
    if scale1 == 0. or scale2 == 0.:
        del train[feature_name]
        del dev[feature_name]
        print('Feature %s was dropped because it has no variance in train or dev' % feature_name)
    else:
        print('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift1).astype(np.float64) / scale1
        dev[feature_name] = (dev[feature_name] - shift2).astype(np.float64) / scale2

    return train, dev


def preprocesar_datos_texto(texto):
    # Tokenización y limpieza básica
    tokens = nltk.word_tokenize(texto)
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Convertir a minúsculas y quitar no alfabéticos

    # Eliminación de stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstrucción del texto preprocesado
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def comprobar_modelo(clf, devX, devY):
    # PRE: clf generado del metodo, KNN, decisiontree o random forest y sus respectivos dev
    # POST: su valor de f1-score, recall, precision con macro avg o weighted avg en función de lo elegído
    predictions = clf.predict(devX)
    report = classification_report(devY, predictions, output_dict=True, zero_division=0)
    if AVG == "macro":
        metric_value = report["macro avg"][BONANZA]
    elif AVG == "weighted":
        metric_value = report["weighted avg"][BONANZA]
    metric_value = float(metric_value)

    return metric_value


def guardar_modelo(nombre, clf):
    # PRE: nombre del .sav a crear y clf generado del metodo, KNN, decisiontree o random forest y sus respectivos dev
    # POST: se crea el .sav
    file_path = os.path.join(OUTPUT_FILE, nombre)
    saved_model = pickle.dump(clf, open(file_path, 'wb'))
    print(f'Modelo {nombre} guardado correctamente')

def guardar_resultados_global():
    # PRE: ---
    # POST: se crea el .csv con los resultados obtenidos
    df_resultados = pd.DataFrame(RESULTADOS_MODELO)
    df_resultados.to_csv(os.path.join(OUTPUT_FILE, "resultados_modelos.csv"), index=False)
    print("Resultados de todos los modelos guardados correctamente.")


def crear_directorio_modelos():
    # PRE: ---
    # POST: Se crea el directorio en el que guardar los archivos generados
    if not os.path.isdir(OUTPUT_FILE):
        os.mkdir(OUTPUT_FILE)
    # Borramos el contenido del directorio
    for filename in os.listdir(OUTPUT_FILE):
        file_path = os.path.join(OUTPUT_FILE, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def metodo_knn(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos KNN con los valores introducidos y guarda el mejor de ellos
    mejor_nombre = ""
    mejor_valor = 0
    for d in D:
        for k in range(K_MIN, K_MAX + 1, K_AUMENTO):
            for p in range(P_MIN, P_MAX + 1, P_AUMENTO):
                nombre_modelo = f"KNN-k:{k}-p:{p}-d:{d}-A:{AVG}-B:{BONANZA}"
                print(nombre_modelo)

                clf = KNeighborsClassifier(n_neighbors=k,
                                           weights=d,
                                           algorithm='auto',
                                           leaf_size=30,
                                           p=p)
                # Indica que las clases están balanceadas -> paso anterior de undersampling
                clf.class_weight = "balanced"
                # Ajustamos el modelo a nuestro train
                clf.fit(trainX, trainY)

                # Evaluamos el modelo
                metric_value = comprobar_modelo(clf, devX, devY)

                if metric_value > mejor_valor:
                    mejor_valor = metric_value
                    mejor_nombre = nombre_modelo
                    mejor_clf = clf

                RESULTADOS_MODELO.append({'Nombre': nombre_modelo, BONANZA: metric_value})

                predictions = clf.predict(devX)
                predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
                report = classification_report(devY, predictions, output_dict=True, zero_division=0)
                report['modelo'] = nombre_modelo
                df = pd.DataFrame(report).transpose()
                print(df)

    guardar_modelo(f"{mejor_nombre}.sav", mejor_clf)

    return mejor_nombre, mejor_valor


def metodo_decisiontree(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos decision tree con los valores introducidos y guarda el mejor de ellos

    mejor_nombre = ""
    mejor_valor = 0
    for max_depth in range(MIN_DEPTH, MAX_DEPTH + 1, DEPTH_AUMENTO):
        for min_samples_leaf in range(MIN_SAMPLE_LEAF, MAX_SAMPLE_LEAF + 1, SAMPLE_LEAF_AUMENTO):
            nombre_modelo = f"DecisionTree-d:{max_depth}-sl:{min_samples_leaf}-A:{AVG}-B:{BONANZA}"
            print(nombre_modelo)

            clf = DecisionTreeClassifier(random_state=RANDOM_STATE,
                                        criterion=CRITERION,
                                        splitter='best',
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf
            )
            # Indica que las clases están balanceadas -> paso anterior de undersampling
            clf.class_weight = "balanced"
            # Ajustamos el modelo a nuestro train
            clf.fit(trainX, trainY)

            # Evaluamos el modelo
            metric_value = comprobar_modelo(clf, devX, devY)

            if metric_value > mejor_valor:
                mejor_valor = metric_value
                mejor_nombre = nombre_modelo
                mejor_clf = clf

            RESULTADOS_MODELO.append({'Nombre': nombre_modelo, BONANZA: metric_value})

            predictions = clf.predict(devX)
            predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
            report = classification_report(devY, predictions, output_dict=True, zero_division=0)
            report['modelo'] = nombre_modelo
            df = pd.DataFrame(report).transpose()
            print(df)

    guardar_modelo(f"{mejor_nombre}.sav", mejor_clf)

    return mejor_nombre, mejor_valor

def metodo_randomforest(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos random forest con los valores introducidos y guarda el mejor de ellos

    mejor_nombre = ""
    mejor_valor = 0
    for n_estimators in range(MIN_N_ESTIMATOR, MAX_N_ESTIMATOR + 1, N_ESTIMATOR_AUMENTO):
        for max_depth in range(MIN_DEPTH, MAX_DEPTH + 1, DEPTH_AUMENTO):
            for min_samples_leaf in range(MIN_SAMPLE_LEAF, MAX_SAMPLE_LEAF + 1, SAMPLE_LEAF_AUMENTO):
                nombre_modelo = f"RandomForest-n:{n_estimators}-d:{max_depth}-sl:{min_samples_leaf}-A:{AVG}-B:{BONANZA}"
                print(nombre_modelo)

                clf = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=RANDOM_STATE,
                )

                # Ajustamos el modelo a nuestro train
                clf.fit(trainX, trainY)

                # Evaluamos el modelo
                metric_value = comprobar_modelo(clf, devX, devY)

                if metric_value > mejor_valor:
                    mejor_valor = metric_value
                    mejor_nombre = nombre_modelo
                    mejor_clf = clf

                RESULTADOS_MODELO.append({'Nombre': nombre_modelo, BONANZA: metric_value})

                predictions = clf.predict(devX)
                predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
                report = classification_report(devY, predictions, output_dict=True, zero_division=0)
                report['modelo'] = nombre_modelo
                df = pd.DataFrame(report).transpose()
                print(df)

    guardar_modelo(f"{mejor_nombre}.sav", mejor_clf)

    return mejor_nombre, mejor_valor

#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main():
    # Entrada principal del programa
    print("---- Iniciando main...")
    crear_directorio_modelos()
    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT_FILE)
    
    # Seleccionamos atributos son relevantes para la clasificación
    if CUANTOS_ATRIBUTOS_SELECCIONAS == "todos":
        atributos = ml_dataset.columns
    elif CUANTOS_ATRIBUTOS_SELECCIONAS == "pocos":
        atributos = CUALES_DESCARTAS_O_ELIGES
    elif CUANTOS_ATRIBUTOS_SELECCIONAS =="menos":
        atributos = atributos_excepto(ml_dataset.columns, CUALES_DESCARTAS_O_ELIGES)
    
    imprimir_atributos(atributos) # Mostramos los atributos elegidos

    # De todo el conjunto de datos nos quedamos con aquellos atributos relevantes
    ml_dataset = ml_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    numerical_features = []
    categorical_features = []
    text_features = []

    categorical_threshold = 0.2

    # Dividimos las columnas entre numericas, categoriales y textuales
    for column in ml_dataset.columns:
        unique_count = ml_dataset[column].nunique()
        total_count = len(ml_dataset[column])

        # Si todos los valores en la columna son numéricos, considerarla numérica
        if pd.api.types.is_numeric_dtype(ml_dataset[column]):
            numerical_features.append(column)
        # Si la columna no es numérica pero tiene una baja cardinalidad, considerarla categórica
        elif unique_count / total_count < categorical_threshold:
            categorical_features.append(column)
        # De lo contrario, considerarla textual
        else:
            text_features.append(column)
    
    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)
    categorical_features = [feature for feature in categorical_features if feature in ml_dataset.columns]
    ml_dataset[categorical_features] = ml_dataset[categorical_features].apply(lambda x: pd.factorize(x)[0])

    print("---- Tratamos el TARGET: " + TARGET_NAME)
    # Creamos la columna __target__ con el atributo a predecir
    unique_targets = ml_dataset[TARGET_NAME].unique()
    target_map = {target: index for index, target in enumerate(unique_targets)}
    ml_dataset['__target__'] = ml_dataset[TARGET_NAME].map(str).map(target_map)
    ml_dataset['__target__'] = ml_dataset[TARGET_NAME]

    del ml_dataset[TARGET_NAME]
    
    # Borramos aquellas entradas de datos en las que el target sea null
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    
    print("---- Dataset empleado")
    print(ml_dataset.head(5))

    # Se decide si hacer o no el preprocesado en funcion de la variable asignada por el usuario
    if HACER_PREPROCESADO:
        print("---- Se preprocesan los datos")
        for columna in text_features:
            ml_dataset[columna] = ml_dataset[columna].apply(preprocesar_datos_texto)
        drop_rows_when_missing=[]
        impute_when_missing=obtener_lista_impute_para(ml_dataset.columns,IMPUTAR_CON,["__target__"])
        rescale_features=obtener_lista_rescalado_para(ml_dataset.columns, METODO_REESCALADO, ["__target__"])
        train, dev = preprocesar_datos(ml_dataset,drop_rows_when_missing,impute_when_missing,rescale_features)

    else:
        print("---- No se preprocesan los datos")
        train, dev = train_test_split(ml_dataset,test_size=DEV_SIZE,random_state=RANDOM_STATE,stratify=ml_dataset[['__target__']])

    print("---- Dataset preprocesado")
    print("TRAIN: ")
    print(train.head(5))
    print(train['__target__'].value_counts())
    print("DEV: ")
    print(dev.head(5))
    print(dev['__target__'].value_counts())
    
    trainX = train.drop('__target__', axis=1)
    devX = dev.drop('__target__', axis=1)
    trainY = np.array(train['__target__'])
    devY = np.array(dev['__target__'])
    
    print("---- Undersampling... o Oversampling??")
    if UNDERSAMPLING_OVERSAMPLING == "undersampling":
        if len(target_map) == 2:
            undersample = RandomUnderSampler(sampling_strategy=0.5)
        else:
            undersample = RandomUnderSampler(sampling_strategy="not minority")

        trainX, trainY = undersample.fit_resample(trainX, trainY)
        devX, devY = undersample.fit_resample(devX, devY)

    elif UNDERSAMPLING_OVERSAMPLING == "oversampling":
        if len(target_map) == 2:
            oversample = RandomOverSampler(sampling_strategy=0.5)
        else:
            oversample = RandomOverSampler(sampling_strategy="not minority")

        trainX, trainY = oversample.fit_resample(trainX, trainY)
        devX, devY = oversample.fit_resample(devX, devY)

    print("---- Iniciando barrido de parámetros ")
    print("TRAINX: ")
    print(trainX.head(5))
    print("DEVX: ")
    print(devX.head(5))

    if ALGORITMO_A_USAR == "KNN":
        mejor_nombre, mejor_valor = metodo_knn(trainX, trainY, devX, devY)
    if ALGORITMO_A_USAR == "decisiontree":
        mejor_nombre, mejor_valor = metodo_decisiontree(trainX, trainY, devX, devY)
    if ALGORITMO_A_USAR == "randomforest":
        mejor_nombre, mejor_valor = metodo_randomforest(trainX, trainY, devX, devY)


    print("El mejor modelo " + ALGORITMO_A_USAR + " según " + str(BONANZA) + " es " + mejor_nombre + " con un valor de " + str(mejor_valor))


if __name__ == "__main__":


    # Registramos la configuración del script
    load_options()
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    main()
    guardar_resultados_global()
