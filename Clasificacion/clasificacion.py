import os
import shutil
from sys import exit, version_info
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

INPUT_FILE      = "Modified_AirlinesReviews.csv"                    # Path del archivo de entrada
TARGET_NAME     = "rating"                      # Nombre de la columna a clasificar
DEV_SIZE        = 0.2                           # Indice del tamaño del dev. Por defecto un 20% de la muestra
RANDOM_STATE    = 42                            # Seed del random split
METODO_REESCALADO = "MIN/MAX"
HACER_PREPROCESADO = True
METODO_PREPROCESADO = "Impute"
CUANTOS_ATRIBUTOS_SELECCIONAS = "todos"
CUALES_DESCARTAS_O_ELIGES = ["Largo de sepalo"]
UNDERSAMPLING_OVERSAMPLING = "undersampling"
COLUMNA_FILA_ELEGIDA = "Airline"
FILA_ELEGIDA = "Emirates"
RESULTADOS_MODELO = []
SAMPLERATE=1
MODO= "entrenar"
RUTA_MODELO= "Clasificacion\\output\\NaiveBayes.sav"
N_GRAMAS = "unigramas"
ALGORITMO = "NaiveBayes"

#Naive Bayes
ALPHA = 1.0

#Logistic Regression
C = 1.0
MAX_ITER = 100
SOLVER = 'lbfgs'
PENALTY = 'l2'
MULTI_CLASS = 'auto'

#Linear SVM
#C = 1.0
#PENALTY = 'l2'
#MAX_ITER = 100
DUAL = False
LOSS = 'squared_hinge'

#XGBoost
MAX_DEPTH = 6
LEARNING_RATE = 0.3
N_ESTIMATORS = 100
GAMMA = 0.0
MIN_CHILD_WEIGHT = 1

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: entrenar_knn.py <optional-args>")
    print("In order to use this pyhton program, you need to feel config.json with the following settings:")
    print(f"INPUT_FILE     input file path of the data                 DEFAULT: ./{INPUT_FILE}")
    print(f"TARGET_NAME    name of the target variable                DEFAULT: ./{TARGET_NAME}")
    print(f"DEV_SIZE        index of size of the dev          DEFAULT: {DEV_SIZE}")
    print(f"RANDOM_STATE   random seed of the split         DEFAULT: {RANDOM_STATE}")
    print(f"METODO_REESCALADO     rescale method used        DEFAULT: {METODO_REESCALADO}")
    print(f"HACER_PREPROCESADO     preprocess the data ---> yes|true / no|false        DEFAULT: {HACER_PREPROCESADO}")
    print(f"METODO_PREPROCESADO    shows the method selected ---> Impute/Drop         DEFAULT: {METODO_PREPROCESADO}")
    print(f"CUANTOS_ATRIBUTOS_SELECCIONAS     how many attributes are selected for the model        DEFAULT: {CUANTOS_ATRIBUTOS_SELECCIONAS}")
    print(f"CUALES_DESCARTAS_O_ELIGES     which attributes are discarded or chosen explicitly        DEFAULT: {CUALES_DESCARTAS_O_ELIGES}")
    print(f"UNDERSAMPLING_OVERSAMPLING     whether to perform undersampling or oversampling         DEFAULT: {UNDERSAMPLING_OVERSAMPLING}")
    print(f"SAMPLERATE     value of SAMPLERATE for oversampling and undersampling        DEFAULT: {SAMPLERATE}")
    print(f"COLUMNA_FILA_ELEGIDA     in which column is the line you want to use         DEFAULT: {COLUMNA_FILA_ELEGIDA}")
    print(f"FILA_ELEGIDA     the line you want to use         DEFAULT: {FILA_ELEGIDA}")
    print(f"RESULTADOS_MODELO     where to store the model's results        DEFAULT: {RESULTADOS_MODELO}")
    print(f"MODO     whether to train or predict         DEFAULT: {MODO}")
    print(f"RUTA_MODELO     where thw .sav is        DEFAULT: {RUTA_MODELO}")
    print(f"N_GRAMAS     n-grams to use         DEFAULT: {N_GRAMAS}")
    print(f"ALGORITMO     algorithm to use         DEFAULT: {ALGORITMO}")
    print(f"ALPHA     alpha value for Naive Bayes         DEFAULT: {ALPHA}")
    print(f"C     C value for Logistic Regression or LinearSVM        DEFAULT: {C}")
    print(f"MAX_ITER     max iterations for Logistic Regression or LinearSVM         DEFAULT: {MAX_ITER}")
    print(f"SOLVER     solver for Logistic Regression         DEFAULT: {SOLVER}")
    print(f"PENALTY     penalty for Logistic Regression or LinearSWM        DEFAULT: {PENALTY}")
    print(f"MULTI_CLASS     multi_class for Logistic Regression         DEFAULT: {MULTI_CLASS}")
    print(f"DUAL     dual for LinearSVM         DEFAULT: {DUAL}")
    print(f"LOSS     loss for LinearSVM         DEFAULT: {LOSS}")
    print(f"MAX_DEPTH     max depth for XGBoost         DEFAULT: {MAX_DEPTH}")
    print(f"LEARNING_RATE     learning rate for XGBoost         DEFAULT: {LEARNING_RATE}")
    print(f"N_ESTIMATORS     number of estimators for XGBoost         DEFAULT: {N_ESTIMATORS}")
    print(f"GAMMA     gamma for XGBoost         DEFAULT: {GAMMA}")
    print(f"MIN_CHILD_WEIGHT     min child weight for XGBoost         DEFAULT: {MIN_CHILD_WEIGHT}")
    exit(1)

def load_options():
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_FILE, DEV_SIZE, RANDOM_STATE, LANGUAGE, SAMPLERATE, COLUMNA_FILA_ELEGIDA, FILA_ELEGIDA
    global TARGET_NAME, METODO_REESCALADO, HACER_PREPROCESADO, METODO_PREPROCESADO
    global CUALES_DESCARTAS_O_ELIGES, CUANTOS_ATRIBUTOS_SELECCIONAS, UNDERSAMPLING_OVERSAMPLING, MODO, RUTA_MODELO, N_GRAMAS, ALGORITMO, ALPHA
    global C, MAX_ITER, SOLVER, PENALTY, MULTI_CLASS, DUAL, LOSS, MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, GAMMA, MIN_CHILD_WEIGHT

    if len(sys.argv) > 1:

        if '-h' in sys.argv or '--help' in sys.argv:
            usage()
        else:
            config_path = sys.argv[1]
            with open(config_path, 'r') as config_file:
                json_config = json.load(config_file)

            if 'INPUT_FILE' in json_config:
                INPUT_FILE = json_config['INPUT_FILE']
            else:
                print("Valor no válido para INPUT_FILE; usando el valor predeterminado.")
            

            if 'DEV_SIZE' in json_config and isinstance(json_config['DEV_SIZE'], float):
                DEV_SIZE = json_config['DEV_SIZE']
            else:
                print("Valor no válido para DEV_SIZE; usando el valor predeterminado.")

            if 'RANDOM_STATE' in json_config and isinstance(json_config['RANDOM_STATE'], int):
                RANDOM_STATE = json_config['RANDOM_STATE']
            else:
                print("Valor no válido para RANDOM_STATE; usando el valor predeterminado.")

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

            if 'METODO_REESCALADO' in json_config and json_config['METODO_REESCALADO'] in ['MIN/MAX','Z-SOCRE']:
                METODO_REESCALADO = json_config['METODO_REESCALADO']
            else:
                print("Valor no válido para METODO_REESCALADO; usando el valor predeterminado.")

            if 'CUANTOS_ATRIBUTOS_SELECCIONAS' in json_config :
                CUANTOS_ATRIBUTOS_SELECCIONAS = json_config["CUANTOS_ATRIBUTOS_SELECCIONAS"]
            else:
                print("Valor no encontrado para CUANTOS_ATRIBUTOS_SELECCIONAS; usando el valor predeterminado.")

            if 'CUALES_DESCARTAS_O_ELIGES' in json_config :
                CUALES_DESCARTAS_O_ELIGES = json_config["CUALES_DESCARTAS_O_ELIGES"]
            else:
                print("Valor no encontrado para CUALES_DESCARTAS_O_ELIGES; usando el valor predeterminado.")

            if 'COLUMNA_FILA_ELEGIDA' in json_config :
                COLUMNA_FILA_ELEGIDA = json_config["COLUMNA_FILA_ELEGIDA"]
            else:
                print("Valor no encontrado para COLUMNA_FILA_ELEGIDA; usando el valor predeterminado.")

            if 'FILA_ELEGIDA' in json_config :
                FILA_ELEGIDA = json_config["FILA_ELEGIDA"]
            else:
                print("Valor no encontrado para FILA_ELEGIDA; usando el valor predeterminado.")

            if 'UNDERSAMPLING_OVERSAMPLING' in json_config and json_config['UNDERSAMPLING_OVERSAMPLING'] in ['undersampling','oversampling','oversamplingsmote']:
                UNDERSAMPLING_OVERSAMPLING = json_config["UNDERSAMPLING_OVERSAMPLING"]
            else:
                print("Valor no válido para UNDERSAMPLING_OVERSAMPLING; usando el valor predeterminado.")

            if 'SAMPLERATE' in json_config and isinstance(json_config['SAMPLERATE'], float):
                SAMPLERATE = json_config['SAMPLERATE']
            else:
                print("Valor no válido para SAMPLERATE; usando el valor predeterminado.")

            if 'LANGUAGE' in json_config :
                LANGUAGE = json_config["LANGUAGE"]
            else:
                print("Valor no encontrado para LANGUAGE; usando el valor predeterminado.")
            if 'MODO' in json_config and json_config['MODO'] in ['entrenar','predecir']:
                MODO = json_config["MODO"]
            else: 
                print("Valor no válido para MODO; usando el valor predeterminado.")
            if 'RUTA_MODELO' in json_config:

                RUTA_MODELO = json_config["RUTA_MODELO"]
            else:
                print("Valor no encontrado para RUTA_MODELO; usando el valor predeterminado.")

            if 'N_GRAMAS' in json_config and json_config['N_GRAMAS'] in ['unigramas','bigramas','trigramas']:
                N_GRAMAS = json_config["N_GRAMAS"]
            else:
                print("Valor no válido para N_GRAMAS; usando el valor predeterminado.")
            if 'ALGORITMO' in json_config and json_config['ALGORITMO'] in ['NaiveBayes', 'LogisticRegression', 'LinearSVM', 'XGBoost']:
                ALGORITMO = json_config["ALGORITMO"]
            else:
                print("Valor no válido para ALGORITMO; usando el valor predeterminado.")

            if 'ALPHA' in json_config and isinstance(json_config['ALPHA'], float):
                ALPHA = json_config['ALPHA']
            else:
                print("Valor no válido para ALPHA; usando el valor predeterminado.")
            if 'C' in json_config and isinstance(json_config['C'], float):
                C = json_config['C']
            else: 
                print("Valor no válido para C; usando el valor predeterminado.")
            if 'MAX_ITER' in json_config and isinstance(json_config['MAX_ITER'], int):
                MAX_ITER = json_config['MAX_ITER']
            else: 
                print("Valor no válido para MAX_ITER; usando el valor predeterminado.")
            if 'SOLVER' in json_config:
                SOLVER = json_config['SOLVER']
            else:
                print("Valor no encontrado para SOLVER; usando el valor predeterminado.")
            if 'PENALTY' in json_config:
                PENALTY = json_config['PENALTY']
            else:
                print("Valor no encontrado para PENALTY; usando el valor predeterminado.")
            if 'MULTI_CLASS' in json_config:
                MULTI_CLASS = json_config['MULTI_CLASS']
            else:
                print("Valor no encontrado para MULTI_CLASS; usando el valor predeterminado.")
            if 'DUAL' in json_config and isinstance(json_config['DUAL'], bool):
                DUAL = json_config['DUAL']
            else:
                print("Valor no válido para DUAL; usando el valor predeterminado.")
            if 'LOSS' in json_config and json_config['LOSS'] in ['squared_hinge', 'hinge']:
                LOSS = json_config['LOSS']
            else:
                print("Valor no encontrado para LOSS; usando el valor predeterminado.")

            if 'MAX_DEPTH' in json_config and isinstance(json_config['MAX_DEPTH'], int):
                MAX_DEPTH = json_config['MAX_DEPTH']
            else:
                print("Valor no válido para MAX_DEPTH; usando el valor predeterminado.")
            
            if 'LEARNING_RATE' in json_config and isinstance(json_config['LEARNING_RATE'], float):
                LEARNING_RATE = json_config['LEARNING_RATE']
            else:
                print("Valor no válido para LEARNING_RATE; usando el valor predeterminado.")

            if 'N_ESTIMATORS' in json_config and isinstance(json_config['N_ESTIMATORS'], int):
                N_ESTIMATORS = json_config['N_ESTIMATORS']
            else:
                print("Valor no válido para N_ESTIMATORS; usando el valor predeterminado.")
            
            if 'GAMMA' in json_config and isinstance(json_config['GAMMA'], float):
                GAMMA = json_config['GAMMA']
            else:
                print("Valor no válido para GAMMA; usando el valor predeterminado.")
            
            if 'MIN_CHILD_WEIGHT' in json_config and isinstance(json_config['MIN_CHILD_WEIGHT'], int):
                MIN_CHILD_WEIGHT = json_config['MIN_CHILD_WEIGHT']
            else:
                print("Valor no válido para MIN_CHILD_WEIGHT; usando el valor predeterminado.")

    else:
        exit()
        
def show_script_options():
    # PRE: ---
    # POST: imprimimos la configuración del script
    print("entrenar_knn.py configuration:")
    print(f"INPUT_FILE                        {INPUT_FILE}")
    print(f"TARGET_NAME                       {TARGET_NAME}")
    print(f"DEV_SIZE                          {DEV_SIZE}")
    print(f"RANDOM_STATE                      {RANDOM_STATE}")
    print(f"METODO_REESCALADO                 {METODO_REESCALADO}")
    print(f"HACER_PREPROCESADO                {HACER_PREPROCESADO}")
    print(f"METODO_PREPROCESADO               {METODO_PREPROCESADO}")
    print(f"CUANTOS_ATRIBUTOS_SELECCIONAS     {CUANTOS_ATRIBUTOS_SELECCIONAS}")
    print(f"CUALES_DESCARTAS_O_ELIGES         {CUALES_DESCARTAS_O_ELIGES}")
    print(f"UNDERSAMPLING_OVERSAMPLING        {UNDERSAMPLING_OVERSAMPLING}")
    print(f"SAMPLERATE                        {SAMPLERATE}")
    print(f"COLUMNA_FILA_ELEGIDA              {COLUMNA_FILA_ELEGIDA}")
    print(f"FILA_ELEGIDA                      {FILA_ELEGIDA}")
    print(f"RESULTADOS_MODELO                 {RESULTADOS_MODELO}")
    print(f"MODO                             {MODO}")
    print(f"RUTA_MODELO                      {RUTA_MODELO}")
    print(f"N_GRAMAS                         {N_GRAMAS}")
    print(f"ALGORITMO                        {ALGORITMO}")
    print(f"ALPHA                            {ALPHA}")
    print(f"C                               {C}")
    print(f"MAX_ITER                         {MAX_ITER}")
    print(f"SOLVER                           {SOLVER}")
    print(f"PENALTY                          {PENALTY}")
    print(f"MULTI_CLASS                      {MULTI_CLASS}")
    print(f"DUAL                             {DUAL}")
    print(f"LOSS                             {LOSS}")
    print(f"MAX_DEPTH                        {MAX_DEPTH}")
    print(f"LEARNING_RATE                    {LEARNING_RATE}")
    print(f"N_ESTIMATORS                     {N_ESTIMATORS}")
    print(f"GAMMA                            {GAMMA}")
    print(f"MIN_CHILD_WEIGHT                  {MIN_CHILD_WEIGHT}")
    print("")


def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)

    # Si no es anterior a la version 3 de python
    return str(x)

def crear_directorio_modelos():
    # PRE: ---
    # POST: Se crea el directorio en el que guardar los archivos generados

    dir_path = os.path.join(f"./output/{ALGORITMO}")
    os.makedirs(dir_path, exist_ok=True)  # Creará el directorio si no existe


def atributos_excepto(atributos, excepciones):
    # PRE: lista completa de atributos y lista de aquellos que no queremos seleccionar
    # POST: devolvemos una lista de atributos
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)
            print(atribs)

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

def borrar_faltantes(df):
    df_sin_faltantes = df.dropna()

    # Verificar si se eliminaron filas con valores faltantes
    if df_sin_faltantes.equals(df):
        print("No hay valores faltantes en el datasheet.")
    else:
        print("Se han eliminado filas con valores faltantes.")
    print(df_sin_faltantes)
    return df_sin_faltantes

def rellenar_valores_faltantes(ml_dataset, numerical_features, categorical_features):
    print('--Rellenamos los valores faltantes con la media o la moda, dependiendo si son numéricos o categóricos--')
    imprimir_faltantes(ml_dataset)
    if numerical_features:
        ml_dataset[numerical_features] = ml_dataset[numerical_features].fillna(ml_dataset[numerical_features].mean())
    if categorical_features:
        ml_dataset[categorical_features] = ml_dataset[categorical_features].fillna(ml_dataset[categorical_features].mode().iloc[0])
    return ml_dataset

def imprimir_faltantes(df):
    # Encontrar las filas con al menos un valor faltante
    filas_con_nan = df[df.isnull().any(axis=1)]

    # Verificar la cantidad de filas encontradas
    if filas_con_nan.empty:
        print("No hay valores faltantes en el datasheet.")
    else:
        # Limitar el resultado a las primeras 5 filas, si hay más de 5
        filas_a_imprimir = filas_con_nan.head(5)
        print(filas_a_imprimir)

def undersample_df(df, target):
    # Determinar el número de instancias en la clase menos representada
    class_counts = df[target].value_counts()
    max_class_size = class_counts.max()
    min_class_size = class_counts.min()

    target_size = int(max_class_size - (max_class_size - min_class_size) * SAMPLERATE)

    # Crear un DataFrame vacío para almacenar el resultado del undersampling
    df_balanced = pd.DataFrame(columns=df.columns)

    # Realizar undersampling para cada clase
    for class_label in df[target].unique():
        df_class = df[df[target] == class_label]
        if len(df_class) > target_size:
            df_class_under = resample(df_class,
                                      replace=False, # Sin reemplazo para evitar duplicados
                                      n_samples=target_size, # Nuevo tamaño basado en la tasa de undersampling
                                      random_state=123) # Para reproducibilidad
        else:
            df_class_under = df_class
        df_balanced = df_balanced._append(df_class_under, ignore_index=True)

    return df_balanced

def oversample_df(df, target):
    # Determinar el número de instancias en la clase más representada y menos representada
    class_counts = df[target].value_counts()
    max_class_size = class_counts.max()
    min_class_size = class_counts.min()

    # Calcular el tamaño objetivo para las clases minoritarias
    # Incrementar el tamaño de las clases minoritarias por un factor del tamaño de la clase mayoritaria
    target_size = int(min_class_size + (max_class_size - min_class_size) * SAMPLERATE)

    # Crear un DataFrame vacío para almacenar el resultado del oversampling
    df_balanced = pd.DataFrame(columns=df.columns)

    # Realizar oversampling para cada clase
    for class_label in df[target].unique():
        df_class = df[df[target] == class_label]
        if len(df_class) < target_size:
            df_class_over = resample(df_class,
                                     replace=True,
                                     # con reemplazo, así se podrán repetir las instancias y aumentar el tamaño de la clase
                                     n_samples=target_size,
                                     random_state=123)
        else:
            df_class_over = df_class
        df_balanced = df_balanced._append(df_class_over, ignore_index=True)
    return df_balanced

def oversample_dataframe(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    oversample = RandomOverSampler(sampling_strategy='auto')
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    return df_resampled

def scale_column(df, column_name):
    # Copiar el DataFrame para evitar modificar el original
    df_scaled = df.copy()

    if METODO_REESCALADO == "MIN/MAX":  # Escalado Min-Max
        # Obtener los valores mínimos y máximos de la columna especificada
        min_value = df[column_name].min()
        max_value = df[column_name].max()

        # Aplicar la fórmula de escalado Min-Max a la columna
        df_scaled[column_name] = (df[column_name] - min_value) / (max_value - min_value)

    elif METODO_REESCALADO == "Z-SCORE":   # Escalado Z-Score
        # Calcular la media y la desviación estándar de la columna especificada
        mean_value = df[column_name].mean()
        std_dev = df[column_name].std()

        # Aplicar la fórmula de escalado Z-Score a la columna
        df_scaled[column_name] = (df[column_name] - mean_value) / std_dev

    return df_scaled


def preprocesar_datos(df, numerical_features, categorical_features, text_features):
    print("Dimensiones de df antes de la transformación: ", df.shape)
    
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists('output'):
        os.makedirs('output')

    # Borrar o imputar valores faltantes
    if METODO_PREPROCESADO == "Drop":
        df = borrar_faltantes(df)
    elif METODO_PREPROCESADO == "Impute":
        df = rellenar_valores_faltantes(df, numerical_features, categorical_features)

    if MODO == "entrenar":
        print("---- Undersampling... o Oversampling??")
        if UNDERSAMPLING_OVERSAMPLING == "undersampling":
            print("antes")
            print(df['__target__'].value_counts())
            df = undersample_df(df, '__target__')
            print("despues")
            print(df['__target__'].value_counts())
        elif UNDERSAMPLING_OVERSAMPLING == "oversampling":
            print("antes")
            print(df['__target__'].value_counts())
            df = oversample_df(df, '__target__')
            print("despues")
            print(df['__target__'].value_counts())
        elif UNDERSAMPLING_OVERSAMPLING == "oversamplingsmote":
            print("antes")
            print(df['__target__'].value_counts())
            df = oversample_dataframe(df, '__target__')
            print("despues")
            print(df['__target__'].value_counts())

    # Reescalar los datos según el parámetro introducido
    for column in numerical_features:
        df = scale_column(df, column)

    # Combinar texto en una sola columna
    df['combined_text'] = df[text_features].astype(str).apply(lambda x: ' '.join(x), axis=1)

    # Modificar esta sección para configurar el rango de n-gramas
    if N_GRAMAS == 'unigramas':
        ngram_range = (1, 1)
    elif N_GRAMAS == 'bigramas':
        ngram_range = (2, 2)
    elif N_GRAMAS == 'trigramas':
        ngram_range = (3, 3)
    else:
        raise ValueError("N_GRAMAS debe ser 'unigramas', 'bigramas' o 'trigramas'")

    text_transformer = TfidfVectorizer(ngram_range=ngram_range)

    # Preprocesar texto
    for columna in text_features:
        df[columna] = df[columna].apply(preprocesar_datos_texto)

    imprimir_faltantes(df)
    data_parts = []
    dir_path = os.path.join(f"./output/{ALGORITMO}")
    if text_features:
        # La transformación del texto debe ocurrir aquí.
        if MODO == "entrenar":
            text_transformed = text_transformer.fit_transform(df['combined_text'])
            with open(os.path.join(dir_path, 'text_transformer.pkl'), 'wb') as f:
                pickle.dump(text_transformer, f)
        else:
            with open(os.path.join(dir_path, 'text_transformer.pkl'), 'rb') as f:
                text_transformer = pickle.load(f)
            text_transformed = text_transformer.transform(df['combined_text'])

        text_dense = text_transformed.toarray() if sp.issparse(text_transformed) else text_transformed
        data_parts.append(text_dense)

    if categorical_features:
        # Codificación OneHot para variables categóricas
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
        if MODO == "entrenar":
            cat_transformed = cat_transformer.fit_transform(df[categorical_features].fillna('missing'))
            with open(os.path.join(dir_path, 'cat_transformer.pkl'), 'wb') as f:
                pickle.dump(cat_transformer, f)
        else:
            with open(os.path.join(dir_path, 'cat_transformer.pkl'), 'rb') as f:
                cat_transformer = pickle.load(f)
            cat_transformed = cat_transformer.transform(df[categorical_features].fillna('missing'))

        cat_dense = cat_transformed.toarray() if sp.issparse(cat_transformed) else cat_transformed
        data_parts.append(cat_dense)

    if numerical_features:
        # Escalado de características numéricas
        num_transformer = StandardScaler()
        if MODO == "entrenar":
            df[numerical_features] = num_transformer.fit_transform(df[numerical_features].fillna(0))
            with open(os.path.join(dir_path, 'num_transformer.pkl'), 'wb') as f:
                pickle.dump(num_transformer, f)
        else:
            with open(os.path.join(dir_path, 'num_transformer.pkl'), 'rb') as f:
                num_transformer = pickle.load(f)
            df[numerical_features] = num_transformer.transform(df[numerical_features].fillna(0))

        num_dense = df[numerical_features]
        data_parts.append(num_dense)
    
    if data_parts:
        combined_data = np.hstack(data_parts)
        print("Combined data shape:", combined_data.shape)
    else:
        print("----------------------------------------------------------------------------------------")
        print("No se han encontrado datos para combinar.")
        print("----------------------------------------------------------------------------------------")
        return 1
    
    if MODO == "entrenar":
        train, dev = train_test_split(np.column_stack((combined_data, df['__target__'].values)), test_size=DEV_SIZE, random_state=RANDOM_STATE, stratify=df['__target__'])
        print("Train shape:", train.shape)
        print("Dev shape:", dev.shape)
        return train, dev
    elif MODO == "predecir":
        return np.column_stack((combined_data, df['__target__'].values))


def target_a_numerico(df, target):
    # PRE: df y target
    # POST: se convierte el target en un valor numérico
    target_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
    # Convertimos el target en un valor numérico
    df['__target__'] = df[target].map(target_map)
    # Damos información sobre el target
    print('Las reseñas positivas son:', (round(df['__target__'].value_counts()[2])),'i.e.', round(df['__target__'].value_counts()[2]/len(df) * 100,2), '% of the dataset')
    print('Las reseñas negativas son:', (round(df['__target__'].value_counts()[0])),'i.e.',round(df['__target__'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
    print('Las reseñas neutrales son:', (round(df['__target__'].value_counts()[1])),'i.e.', round(df['__target__'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
    return df

def preprocesar_datos_texto(texto):
    # Tokenización y limpieza básica
    tokens = nltk.word_tokenize(texto)
    tokens = [token.lower() for token in tokens if token.isalpha()]  # Convertir a minúsculas y quitar no alfabéticos

    # Eliminación de stopwords
    stop_words = set(stopwords.words(LANGUAGE))
    tokens = [token for token in tokens if token not in stop_words]

    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstrucción del texto preprocesado
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def guardar_modelo(nombre, clf):
    
    # POST: se crea el .sav
    dir_path = os.path.join(f"./output/{ALGORITMO}")
    
    
    file_path = os.path.join(dir_path, nombre)
    saved_model = pickle.dump(clf, open(file_path, 'wb'))
    print(f'Modelo {nombre} guardado correctamente')

def imprimir_resultados(predictions, devY_encoded, nombre_modelo, clf):
    # Calcular la matriz de confusión y las métricas de clasificación
    conf_mat = confusion_matrix(devY_encoded, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(devY_encoded, predictions, average=None)
    # Calculando métricas promedio
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(devY_encoded, predictions, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(devY_encoded, predictions, average='micro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(devY_encoded, predictions, average='weighted')
    accuracy = accuracy_score(devY_encoded, predictions)
    metrics_table = [
        ['Average Type', 'Precision', 'Recall', 'F1-Score'],
        ['Micro', precision_micro, recall_micro, f1_micro],
        ['Macro', precision_macro, recall_macro, f1_macro],
        ['Weighted', precision_weighted, recall_weighted, f1_weighted]
        ]
    # Crear la ruta del archivo de texto
    dir_path = os.path.join(f"./output/{ALGORITMO}")
    
    resultados_path = os.path.join(dir_path, 'resultadoTrain.txt')

    # Escribir los resultados en un archivo de texto
    with open(resultados_path, 'w') as f:
        f.write('Matriz de confusion:\n')
        f.write(tabulate(conf_mat, headers=['Predicted negative', 'Predicted neutral', 'Predicted positive'], showindex=['Actual negative', 'Actual neutral', 'Actual positive'], tablefmt='grid'))
        f.write('\n\nPrecision:\n')
        f.write(str(precision))
        f.write('\n\nRecall:\n')
        f.write(str(recall))
        f.write('\n\nF1-score:\n')
        f.write(str(f1))
        f.write('\n\nAccuracy:\n')
        f.write(str(accuracy))
        f.write('\n\nMetricas agregadas:\n')
        f.write(tabulate(metrics_table, headers='firstrow', tablefmt='grid'))
        
        
    report = classification_report(devY_encoded, predictions, output_dict=True, zero_division=0)
    report['modelo'] = nombre_modelo
    df = pd.DataFrame(report).transpose()
    print(df)

    guardar_modelo(f"{nombre_modelo}.sav", clf)


##..................................................................NAIVE BAYES......

def metodo_naiveBayes(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos NaiveBayes con los valores introducidos
    nombre_modelo = f"NaiveBayes"
    print(nombre_modelo)

    label_encoder = LabelEncoder()

    # Ajustar el codificador a las etiquetas y transformarlas a enteros
    trainY_encoded = label_encoder.fit_transform(trainY)
    devY_encoded = label_encoder.transform(devY)
    trainX_non_negative = np.clip(trainX, 0, None)
    devX_non_negative = np.clip(devX, 0, None)
    if np.any(trainX_non_negative < 0):
        print("Aún hay valores negativos en trainX.")
    else:
        print("Todos los valores en trainX son no negativos.")
    clf = MultinomialNB(alpha=ALPHA)
    clf.fit(trainX_non_negative, trainY_encoded)
    predY = clf.predict(devX)

    # Evaluamos el modelo
    predictions = clf.predict(devX_non_negative)
    imprimir_resultados(predictions, devY_encoded, nombre_modelo, clf)
    

def metodo_logisticRegression(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos LogisticRegression con los valores introducidos
    nombre_modelo = f"LogisticRegression"
    print(nombre_modelo)

    label_encoder = LabelEncoder()

    # Ajustar el codificador a las etiquetas y transformarlas a enteros
    trainY_encoded = label_encoder.fit_transform(trainY)
    devY_encoded = label_encoder.transform(devY)
    trainX_non_negative = np.clip(trainX, 0, None)
    devX_non_negative = np.clip(devX, 0, None)
    if np.any(trainX_non_negative < 0):
        print("Aún hay valores negativos en trainX.")
    else:
        print("Todos los valores en trainX son no negativos.")
    
    clf = LogisticRegression(C=C, solver=SOLVER, penalty=PENALTY, max_iter=MAX_ITER, multi_class=MULTI_CLASS)
    clf.fit(trainX_non_negative, trainY_encoded)
    predY = clf.predict(devX)

    # Evaluamos el modelo
    predictions = clf.predict(devX_non_negative)
    
    imprimir_resultados(predictions, devY_encoded, nombre_modelo, clf)

def metodo_linearSVM(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos LinearSVM con los valores introducidos 
    nombre_modelo = f"LinearSVM"
    print(nombre_modelo)

    label_encoder = LabelEncoder()

    # Ajustar el codificador a las etiquetas y transformarlas a enteros
    trainY_encoded = label_encoder.fit_transform(trainY)
    devY_encoded = label_encoder.transform(devY)
    trainX_non_negative = np.clip(trainX, 0, None)
    devX_non_negative = np.clip(devX, 0, None)
    if np.any(trainX_non_negative < 0):
        print("Aún hay valores negativos en trainX.")
    else:
        print("Todos los valores en trainX son no negativos.")
    clf = LinearSVC(C=C, loss=LOSS, penalty=PENALTY, dual=DUAL, max_iter=MAX_ITER)
    clf.fit(trainX_non_negative, trainY_encoded)
    predY = clf.predict(devX)

    # Evaluamos el modelo
    predictions = clf.predict(devX_non_negative)
    imprimir_resultados(predictions, devY_encoded, nombre_modelo, clf)

def metodo_xgboost(trainX, trainY, devX, devY):
    # PRE: trainX, trainY, devX, devY
    # POST: Se generan los modelos XGBoost con los valores introducidos
    nombre_modelo = f"XGBoost"
    print(nombre_modelo)

    label_encoder = LabelEncoder()

    # Ajustar el codificador a las etiquetas y transformarlas a enteros
    trainY_encoded = label_encoder.fit_transform(trainY)
    devY_encoded = label_encoder.transform(devY)
    trainX_non_negative = np.clip(trainX, 0, None)
    devX_non_negative = np.clip(devX, 0, None)
    if np.any(trainX_non_negative < 0):
        print("Aún hay valores negativos en trainX.")
    else:
        print("Todos los valores en trainX son no negativos.")
    clf = XGBClassifier(max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE ,n_estimators=N_ESTIMATORS, gamma=GAMMA, min_child_weight=MIN_CHILD_WEIGHT)
    clf.fit(trainX_non_negative, trainY_encoded)
    predY = clf.predict(devX)

    # Evaluamos el modelo
    predictions = clf.predict(devX_non_negative)
    imprimir_resultados(predictions, devY_encoded, nombre_modelo, clf)

def pasarTargetAString(predictions):
    # PRE: predictions
    # POST: se convierte el target en un valor numérico
    target_map = {0: 'negativo', 1: 'neutro', 2: 'positivo'}
    return [target_map[pred] for pred in predictions]
    
def predecir(modelo, testX, testY):
    modelo = os.path.abspath(modelo)
    with open(modelo, 'rb') as file:
        clf = pickle.load(file)
    
    predictions = clf.predict(testX)
    # Suponiendo que pasarTargetAString() convierte las predicciones numéricas a su forma de cadena correspondiente.
    predictions = pasarTargetAString(predictions)
    
    # Calculamos métricas
    conf_mat = confusion_matrix(testY, predictions)
    report = classification_report(testY, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(testY, predictions, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(testY, predictions, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(testY, predictions, average='micro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(testY, predictions, average='weighted')
    accuracy = accuracy_score(testY, predictions)
    
    metrics_table = [
        ['Average Type', 'Precision', 'Recall', 'F1-Score'],
        ['Micro', precision_micro, recall_micro, f1_micro],
        ['Macro', precision_macro, recall_macro, f1_macro],
        ['Weighted', precision_weighted, recall_weighted, f1_weighted]
    ]

    model_dir = os.path.dirname(modelo)
    
    resultados_path = os.path.join(model_dir, 'resultadosTest.txt')

    with open(resultados_path, 'w') as f:
        f.write('Matriz de confusion:\n')
        f.write(tabulate(conf_mat, headers=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive'], showindex=['Actual Negative', 'Actual Neutral', 'Actual Positive'], tablefmt='grid'))
        f.write('\n\nInforme de clasificacion:\n')
        f.write(report)
        f.write('\n\nMetricas Detalladas:\n')
        f.write(tabulate(metrics_table, headers='firstrow', tablefmt='grid'))
        f.write('\n\nPrecision Detallada:\n')
        f.write(np.array2string(precision))
        f.write('\n\nRecall Detallado:\n')
        f.write(np.array2string(recall))
        f.write('\n\nF1-Score Detallado:\n')
        f.write(np.array2string(f1))
        f.write('\n\nAccuracy:\n')
        f.write(str(accuracy))

    print("Resultados guardados en:", resultados_path)


##..................................................................MAIN......

def main():
    # Entrada principal del program
    print("---- Iniciando main...")

    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT_FILE)
    crear_directorio_modelos()
    if COLUMNA_FILA_ELEGIDA != "none" or FILA_ELEGIDA != "none":
        ml_dataset = ml_dataset[ml_dataset[COLUMNA_FILA_ELEGIDA] == FILA_ELEGIDA]

    # Seleccionamos atributos son relevantes para la clasificación
    if CUANTOS_ATRIBUTOS_SELECCIONAS == "todos":
        atributos = ml_dataset.columns
    elif CUANTOS_ATRIBUTOS_SELECCIONAS == "pocos":
        atributos = CUALES_DESCARTAS_O_ELIGES
    elif CUANTOS_ATRIBUTOS_SELECCIONAS == "menos":
        atributos = atributos_excepto(ml_dataset.columns, CUALES_DESCARTAS_O_ELIGES)

    imprimir_atributos(atributos)  # Mostramos los atributos elegidos

    # De todo el conjunto de datos nos quedamos con aquellos atributos relevantes
    ml_dataset = ml_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    numerical_features = []
    categorical_features = []
    text_features = []

    categorical_threshold = 0.2
    #ml_dataset = target_a_numerico(ml_dataset, TARGET_NAME)
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
    
    # Eliminar la columna objetivo de cada lista de características si está presente
    if TARGET_NAME in numerical_features:
        numerical_features.remove(TARGET_NAME)
    if TARGET_NAME in categorical_features:
        categorical_features.remove(TARGET_NAME)
    if TARGET_NAME in text_features:
        text_features.remove(TARGET_NAME)
        
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



    
    if MODO=="entrenar":
        # Se decide si hacer o no el preprocesado en funcion de la variable asignada por el usuario
        if HACER_PREPROCESADO:
            print("---- Se preprocesan los datos")

            train, dev = preprocesar_datos(ml_dataset, numerical_features, categorical_features, text_features)

        else:
            print("---- No se preprocesan los datos")

            train, dev = train_test_split(ml_dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE,stratify=ml_dataset[['__target__']])
        print(f"------ Iniciando {ALGORITMO} ------")

        # Separar características y etiquetas en los conjuntos de entrenamiento y validación
        trainX = train[:, :-1]  
        trainY = train[:, -1]  
        devX = dev[:, :-1]
        devY = dev[:, -1]
        
        if ALGORITMO == "NaiveBayes":
            metodo_naiveBayes(trainX, trainY, devX, devY)
        elif ALGORITMO == "LogisticRegression":
            metodo_logisticRegression(trainX, trainY, devX, devY)
        elif ALGORITMO == "LinearSVM":
            metodo_linearSVM(trainX, trainY, devX, devY)
        elif ALGORITMO == "XGBoost":
            metodo_xgboost(trainX, trainY, devX, devY)
    
    else:
        # Se decide si hacer o no el preprocesado en funcion de la variable asignada por el usuario
        if HACER_PREPROCESADO:
            print("---- Se preprocesan los datos")

            test = preprocesar_datos(ml_dataset, numerical_features, categorical_features, text_features)

        else:
            test = ml_dataset
        
        # Dividir características y etiquetas
        testX = test[:, :-1]
        testY = test[:, -1]
        print("testY valor:", testY)
        
        # Mostrar dimensiones para verificación
        print("testX shape:", testX.shape)
        print("testY shape:", testY.shape)
        print("------ Iniciando Predicción ------")
        predecir(RUTA_MODELO, testX, testY)

if __name__ == "__main__":
    # Registramos la configuración del script
    load_options()
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    main()