import csv
import sys
from getopt import getopt, GetoptError
from sys import exit, argv, version_info
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from pytz import unicode
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import pickle





#####GENERAL############
OUTPUT = "output.txt"  # Path del archivo de salida
INPUT = "iris.csv"  # Path del archivo de entrada
METODO = 'knn'
TARGET = "Especie"  # Nombre de la columna a clasificar
DEV_SIZE = 0.2  # Indice del tamaño del dev. Por defecto un 20% de la muestra
SEED = 42  # Seed del random split
metrica_seleccionada= "f1"
#####PREPROCESADO############
REMOVEMISSING = 0  #
SAMPLING = 0
SAMPLERATE=1
SCALING = 0
CLEANTEXT = 0
CATEGORICALNUMERICAL = 1

#####KNN############
K_MIN = 2  # Numero minimo de "nearest neighbors"
K_MAX = 3  # Número máximo de "nearest neighbors"
D = ['uniform', 'distance']  # Ponderacion de distancias
P_MIN = 1  # 1=Manhattan 2=Euclídea
P_MAX = 2  # 1=Manhattan 2=Euclídea

######TREE#############
MINDEPTH = 5
MAXDEPTH = 7
MINSAMPLE = 3
MAXSAMPLE = 5

#####RANDOMFOREST##########
MINESTIMATOR=3
MAXESTIMATOR=5
#######################################################################################
#                                 Gestión de argumentos                               #
#######################################################################################
global mejor_metrica_valor

def help():
    print("Usage: entrenar_knn.py <optional-args>")
    print("The options supported by entrenar_knn are:")
    print("-h, --help                  Show the usage")
    print("---------------GENERAL USAGE-------------")
    print(f"-i, --input=<file>         Input file path of the data                 DEFAULT: ./{INPUT}")
    print(f"-o, --output=<file>        Output file path for the weights            DEFAULT: ./{OUTPUT}")
    print(f"-m, --method=<method>     Used method (knn | decisionTree | randomForest)  DEFAULT: {METODO}")
    print(f"--target                  Searched target DEFAULT:{TARGET}")
    print(f"--dev-size                Used dev size            DEFAULT: {DEV_SIZE}")
    print(f"--metric                  Searched metric (precision | recall | f1)             DEFAULT:  {metrica_seleccionada}")
    print("---------------PREPROCESSING-------------")
    print(f"--remove-missing      Remove missing values -> 0: No | 1: Yes | 2 Fill    DEFAULT: {REMOVEMISSING}")
    print(f"--sampling             Sampling type -> 0: No | 1: Undersampling | 2: Oversampling DEFAULT: {SAMPLING}")
    print(f"--scaling             Scaling algorithm-> 0: No | 1: MinMax | 2: Z-score DEFAULT: {SCALING}")
    print(f"--sample-rate         Sample rating                              DEFAULT: {SAMPLERATE}")
    print(f"--clean-text         Clean text -> 0: No | 1: Yes               DEFAULT: {CLEANTEXT}")
    print(f" --categorical-numerical Convert categorical to numerical -> 0: No | 1: Yes DEFAULT: {CATEGORICALNUMERICAL}")
    print(f"---------------KNN ALGORITHM-------------")
    print(f"--distance             Distance parameter                        DEFAULT: {D}")
    print(f"--k-min=<value>            Minimum number of neighbors for KNN        DEFAULT: {K_MIN}")
    print(f"--k-max=<value>            Maximum number of neighbors for KNN        DEFAULT: {K_MAX}")
    print(f"--p-min=<value>            Distance from (1: Manhattan | 2: Euclidean)DEFAULT: {P_MIN}")
    print(f"--p-max=<value>            Distance to (1: Manhattan | 2: Euclidean)  DEFAULT: {P_MAX}")
    print(f"---------------DECISION TREE-------------")
    print(f"--mind=<depth>, --min-depth=<depth> Minimum depth of the tree        DEFAULT: {MINDEPTH}")
    print(f"--maxd=<depth>, --max-depth=<depth> Maximum depth of the tree        DEFAULT: {MAXDEPTH}")
    print(f"--mins=<sample>, --min-sample=<sample> Minimum sample of the tree   DEFAULT: {MINSAMPLE}")
    print(f"--maxs=<sample>, --max-sample=<sample> Maximum sample of the tree   DEFAULT: {MAXSAMPLE}")
    print("---------------RANDOM FOREST-------------")
    print(f"--mind=<depth>, --min-depth=<depth> Minimum depth of the tree        DEFAULT: {MINDEPTH}")
    print(f"--maxd=<depth>, --max-depth=<depth> Maximum depth of the tree        DEFAULT: {MAXDEPTH}")
    print(f"--mins=<sample>, --min-sample=<sample> Minimum sample of the tree   DEFAULT: {MINSAMPLE}")
    print(f"--maxs=<sample>, --max-sample=<sample> Maximum sample of the tree   DEFAULT: {MAXSAMPLE}")
    print(f"--mines=<estimators>, --min-estimator=<estimators> Minimum number of estimators for the forest DEFAULT: {MINESTIMATOR}")
    print(f"--maxes=<estimators>, --max-estimator=<estimators> Maximum number of estimators for the forest DEFAULT: {MAXESTIMATOR}")

    # Salimos del programa
    exit(0)


def load_options(options):
    global OUTPUT, INPUT, METODO, REMOVEMISSING, SAMPLING, \
        SCALING, SAMPLERATE, CLEANTEXT, CATEGORICALNUMERICAL, \
        D, K_MIN, K_MAX, P_MIN, P_MAX, MINDEPTH, MAXDEPTH, \
        MINSAMPLE, MAXSAMPLE, MINESTIMATOR, MAXESTIMATOR, SEED,DEV_SIZE,metrica_seleccionada,TARGET

    for opt, arg in options:
        if opt in ("-h", "--help"):
            help()
        elif opt in ("-i", "--input"):
            INPUT = arg
        elif opt in ("-o", "--output"):
            OUTPUT = arg
        elif opt in ("--target",):
            TARGET = arg
        elif opt in ("-m", "--method"):
            METODO = arg
        elif opt in ('--dev-size'):
            DEV_SIZE=float(arg)
        elif opt in ("--remove-missing",):
            REMOVEMISSING = int(arg)
        elif opt in ("--sampling",):
            SAMPLING = int(arg)
        elif opt in ("--scaling",):
            SCALING = int(arg)
        elif opt in ("--sample-rate",):
            SAMPLERATE = float(arg)
        elif opt in ("--clean-text",):
            CLEANTEXT = int(arg)
        elif opt in ("--categorical-numerical",):
            CATEGORICALNUMERICAL = int(arg)
        elif opt in ("-d", "--distance"):
            D = int(arg)
        elif opt in ("--k-min",):
            K_MIN = int(arg)
        elif opt in ("--k-max",):
            K_MAX = int(arg)
        elif opt in ("--p-min",):
            P_MIN = int(arg)
        elif opt in ("--p-max",):
            P_MAX = int(arg)
        elif opt in ("--min-depth",):
            MINDEPTH = int(arg)
        elif opt in ("--max-depth",):
            MAXDEPTH = int(arg)
        elif opt in ("--min-sample",):
            MINSAMPLE = int(arg)
        elif opt in ("--max-sample",):
            MAXSAMPLE = int(arg)
        elif opt in ("--min-estimator",):
            MINESTIMATOR = int(arg)
        elif opt in ("--max-estimator",):
            MAXESTIMATOR = int(arg)
        elif opt in ("--seed",):
            SEED = int(arg)
        elif opt in ("--metric",):
            metrica_seleccionada=arg


def show_script_options():
    print("Configuration:")
    print("---------------GENERAL USAGE-------------")
    print(f"-i, --input=<file>          Input file path of the data                  : {INPUT}")
    print(f"-o, --output=<file>         Output file path for the weights             : {OUTPUT}")
    print(f"--method=<method>           Used method (knn | decisionTree | randomForest) : {METODO}")
    print(f"--target                  Searched target DEFAULT:{TARGET}")
    print(f"--metric                    Searched metric                        :{metrica_seleccionada}")
    print("---------------PREPROCESSING-------------")
    print(f"--remove-missing            Remove missing values -> 0: No | 1: Yes      : {REMOVEMISSING}")
    print(f"--sampling                  Sampling type -> 0: No | 1: Undersampling | 2: Oversampling : {SAMPLING}")
    print(f"--scaling                   Scaling algorithm-> 0: No | 1: MinMax | 2: Z-score : {SCALING}")
    print(f"--sample-rate               Sample rating                               : {SAMPLERATE}")
    print(f"--clean-text                Clean text -> 0: No | 1: Yes                : {CLEANTEXT}")
    print(f"--categorical-numerical     Convert categorical to numerical -> 0: No | 1: Yes : {CATEGORICALNUMERICAL}")

    if METODO == 'knn':
        print("---------------KNN ALGORITHM-------------")
        print(f"-d, --distance              Distance parameter                         : {D}")
        print(f"--k-min=<value>             Minimum number of neighbors for KNN       : {K_MIN}")
        print(f"--k-max=<value>             Maximum number of neighbors for KNN       : {K_MAX}")
        print(f"--p-min=<value>             Distance from (1: Manhattan | 2: Euclidean) : {P_MIN}")
        print(f"--p-max=<value>             Distance to (1: Manhattan | 2: Euclidean)  : {P_MAX}")
    elif METODO == 'decisionTree':
        print("---------------DECISION TREE-------------")
        print(f"--min-depth=<depth>         Minimum depth of the tree                 : {MINDEPTH}")
        print(f"--max-depth=<depth>         Maximum depth of the tree                 : {MAXDEPTH}")
        print(f"--min-sample=<sample>       Minimum sample size of the tree           : {MINSAMPLE}")
        print(f"--max-sample=<sample>       Maximum sample size of the tree           : {MAXSAMPLE}")
    elif METODO == 'randomForest':
        print("---------------RANDOM FOREST-------------")
        print(f"--min-depth=<depth>         Minimum depth of the tree                 : {MINDEPTH}")
        print(f"--max-depth=<depth>         Maximum depth of the tree                 : {MAXDEPTH}")
        print(f"--min-sample=<sample>       Minimum sample size of the tree           : {MINSAMPLE}")
        print(f"--max-sample=<sample>       Maximum sample size of the tree           : {MAXSAMPLE}")
        print(f"--min-estimator=<estimators> Minimum number of estimators for the forest : {MINESTIMATOR}")
        print(f"--max-estimator=<estimators> Maximum number of estimators for the forest : {MAXESTIMATOR}")

    # Aquí puedes añadir más condiciones si hay otros métodos con opciones específicas
    print(f"--seed                      Seed for random number generation           : {SEED}")


#######################################################################################
#                           Métodos usados por el main                                #
#######################################################################################

def separar_columnas(ml_dataset):
    indice_tipo = []  # 0: numérico, 1: categórico, 2: texto

    for atr in ml_dataset.columns:
        if pd.api.types.is_numeric_dtype(ml_dataset[atr]):
            indice_tipo.append(0)
        elif pd.api.types.is_string_dtype(ml_dataset[atr]):
            indice_tipo.append(1)  # Asignamos 1 a los atributos de texto

    # Identificamos los atributos según el tipo
    atributos_numericos = [atr for i, atr in enumerate(ml_dataset.columns) if indice_tipo[i] == 0]
    atributos_texto = [atr for i, atr in enumerate(ml_dataset.columns) if indice_tipo[i] == 1]

    # Convertimos los atributos categóricos en variables dummy
    # A este punto, ml_dataset_procesado tiene columnas numéricas, dummies para categóricas,
    # y atributos de texto sin cambios. Aquí podrías transformar los textos si es necesario.

    print("Atributos Numéricos:", atributos_numericos)
    print("Atributos de Texto/categoricos:", atributos_texto)

    return atributos_numericos, atributos_texto


import pandas as pd


def separar_texto_categorico(ml_dataset, atributos_texto):
    atributos_categoricos = []

    for atr in atributos_texto:
        # Contamos la frecuencia de cada valor en la columna
        freq_values = ml_dataset[atr].value_counts(normalize=True)

        # Si el valor más común supera el umbral, lo tratamos como categórico
        if freq_values.iloc[1] > 0.2:
            atributos_categoricos.append(atr)

    # Eliminamos los atributos categóricos de la lista de atributos de texto
    atributos_texto = [atr for atr in atributos_texto if atr not in atributos_categoricos]

    # Convertimos atributos categóricos en variables dummy
    ml_dataset = pd.get_dummies(ml_dataset, columns=atributos_categoricos)

    print("Atributos Categóricos:", atributos_categoricos)
    print("Atributos de Texto:", atributos_texto)

    return atributos_texto, atributos_categoricos


def separar_numerico_categorico(ml_dataset, numerical_features, categorical_features):
    for atr in numerical_features:
        if len(ml_dataset[atr].unique()) < 2:
            continue
        # Contamos la frecuencia de cada valor en la columna
        freq_values = ml_dataset[atr].value_counts(normalize=True)

        # Si el valor más común supera el umbral, lo tratamos como categórico
        if freq_values.iloc[1] > 0.3:
            categorical_features.append(atr)

    # Eliminamos los atributos categóricos de la lista de atributos de texto
    numerical_features = [atr for atr in numerical_features if atr not in categorical_features]

    # Convertimos atributos categóricos en variables dummy
    ml_dataset = pd.get_dummies(ml_dataset, columns=categorical_features)

    print("Atributos Categóricos:", categorical_features)
    print("Atributos Numéricos:", numerical_features)

    return numerical_features, categorical_features


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
            entrada = {"feature": a, "impute_with": impute_with}
            lista.append(entrada)

    return lista


def obtener_lista_rescalado_para(atributos, rescale_with, excepciones):
    # PRE: lista de atributos y string indicando con qué valor reescalamos
    # POST: diccionario del estilo: {'num_var45_ult1': 'AVGSTD', ... }

    diccionario = {}
    for a in atributos:
        if a not in excepciones:
            diccionario[a] = rescale_with;

    return diccionario


def normalizar_texto(texto):
    texto = str(texto).lower()  # Convertir a minúsculas
    texto_letras = ''.join(c for c in texto if c.isalpha() or c.isspace())  # Conservar solo letras y espacios
    palabras = word_tokenize(texto_letras)  # Tokenizar
    stop_words = set(stopwords.words('english'))  # Eliminar stopwords, ajusta el idioma si es necesario
    palabras = [p for p in palabras if p.lower() not in stop_words]
    palabras.sort()  # Ordenar alfabéticamente
    stemmer = PorterStemmer()  # Stemming
    palabras = [stemmer.stem(p) for p in palabras]
    return ' '.join(palabras)  # Unir de nuevo en un string


def deCategoricoANumerico(df, categoricos):
    for cat in categoricos:
        # Factorizamos la columna categórica
        categorica_numerica, categorica_numerica_indices = pd.factorize(df[cat])

        # Creamos un nuevo DataFrame con la columna numérica
        df_numerico = pd.DataFrame({f'{cat}': categorica_numerica})

        # Reemplazamos la columna categórica con la columna numérica en el DataFrame original
        df = df.drop(columns=[cat])
        df = pd.concat([df, df_numerico], axis=1)

    # Eliminar las dos últimas filas
    df = df.iloc[:-2, :]
    df.to_csv("nova.csv")
    return df


def undersample_df(df, target, undersample_rate):
    # Determinar el número de instancias en la clase menos representada
    class_counts = df[target].value_counts()
    max_class_size = class_counts.max()
    min_class_size = class_counts.min()

    target_size = int(max_class_size - (max_class_size - min_class_size) * undersample_rate)

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


def oversample_df(df, target, oversample_rate):
    # Determinar el número de instancias en la clase más representada y menos representada
    class_counts = df[target].value_counts()
    max_class_size = class_counts.max()
    min_class_size = class_counts.min()

    # Calcular el tamaño objetivo para las clases minoritarias
    # Incrementar el tamaño de las clases minoritarias por un factor del tamaño de la clase mayoritaria
    target_size = int(min_class_size + (max_class_size - min_class_size) * oversample_rate)

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


def scale_column(df, column_name, scale_type=0):
    # Copiar el DataFrame para evitar modificar el original
    df_scaled = df.copy()

    if scale_type == 1:  # Escalado Min-Max
        # Obtener los valores mínimos y máximos de la columna especificada
        min_value = df[column_name].min()
        max_value = df[column_name].max()

        # Aplicar la fórmula de escalado Min-Max a la columna
        df_scaled[column_name] = (df[column_name] - min_value) / (max_value - min_value)

    elif scale_type == 2:  # Escalado Z-Score
        # Calcular la media y la desviación estándar de la columna especificada
        mean_value = df[column_name].mean()
        std_dev = df[column_name].std()

        # Aplicar la fórmula de escalado Z-Score a la columna
        df_scaled[column_name] = (df[column_name] - mean_value) / std_dev

    return df_scaled


def borrar_faltantes(df):
    df_sin_faltantes = df.dropna()

    # Verificar si se eliminaron filas con valores faltantes
    if df_sin_faltantes.equals(df):
        print("No hay valores faltantes en el datasheet.")
    else:
        print("Se han eliminado filas con valores faltantes.")
    print(df_sin_faltantes)
    return df_sin_faltantes

def preprocesar_datos(df, numerical_features, categorical_features, text_features):  # drop_rows_when_missing, impute_when_missing, rescale_features):
    # PRE: Conjunto completo de datos para ajustar nuestro algoritmo
    # POST: Devuelve dos conjuntos: Train y Dev tratando los missing values y reescalados

    if REMOVEMISSING == 1:
        df = borrar_faltantes(df)
    elif REMOVEMISSING == 2:
        df = rellenar_valores_faltantes(df, numerical_features, categorical_features)

    if SAMPLING == 1:
        print("antes")
        print(df['__target__'].value_counts())
        df = undersample_df(df, '__target__', SAMPLERATE)
        print("despues")
        print(df['__target__'].value_counts())
    elif SAMPLING == 2:
        print("antes")
        print(df['__target__'].value_counts())
        df = oversample_df(df, '__target__', SAMPLERATE)
        print("despues")
        print(df['__target__'].value_counts())

    if SCALING != 0:
        print("----Escalamos los atributos----")
        for column in numerical_features:
            df = scale_column(df,column,SCALING)

    if CLEANTEXT == 1:
        if len(text_features) != 0:
            nltk.download('punkt')
            nltk.download('stopwords')
            for columna in text_features:
                df[columna + '_normalizado'] = df[columna].apply(normalizar_texto)

    imprimir_faltantes(df)

    if CATEGORICALNUMERICAL == 1:
        if len(categorical_features) != 0:
            print("----Pasamos de categoricos a numericos---")
            df = deCategoricoANumerico(df, categorical_features)

    imprimir_faltantes(df)

    # Dividimos nuestros datos de entrenamiento en train y dev
    train, dev = train_test_split(df, test_size=DEV_SIZE, random_state=SEED,
                                  stratify=df[['__target__']])

    return train, dev
    # REESCALAR LOS DATOS
    # se puede hacer un proceso paso a paso como el que se ve aqui o se puede emplear un scaler
    # por ejemplo scaler = StandardScaler()
    # puedo generar un subset del dataframe original copiando en el solo las columnas que son numericas. Recordemos que numeric_columns contiene la lista de los atributos que son numericos
    # df_numeric = df[numeric_columns].copy()
    # df_numeric_sc = pd.DataFrame(scaler.fit_transform(df_numeric))df_numeric_sc.rename(mapper=dict(zip(df_numeric_sc.columns,df_numeric.columns)),axis=1,inplace=True)
    # otro posible scaler es scaler = Normalizer() que escala con respecto al max, min


'''    for (feature_name, rescale_method) in rescale_features.items():
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

'''

def calcular_metrica(y_real, y_pred, metrica):
    """Calcula la métrica especificada."""
    if metrica == 'precision':
        return precision_score(y_real, y_pred, average='macro')
    elif metrica == 'recall':
        return recall_score(y_real, y_pred, average='macro')
    elif metrica == 'f1':
        return f1_score(y_real, y_pred, average='macro')
    else:
        print(f"Métrica {metrica} no reconocida. Usando F1 por defecto.")
        return f1_score(y_real, y_pred, average='macro')

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


def rellenar_valores_faltantes(ml_dataset, numerical_features, categorical_features):
    print('--Rellenamos los valores faltantes con la media o la moda, dependiendo si son numéricos o categóricos--')
    imprimir_faltantes(ml_dataset)
    ml_dataset[numerical_features] = ml_dataset[numerical_features].fillna(ml_dataset[numerical_features].mean())
    ml_dataset[categorical_features] = ml_dataset[categorical_features].fillna(
        ml_dataset[categorical_features].mode().iloc[0])
    return ml_dataset





def comprobar_modelo(modelo, devX, devY, target_map):
    predictions = modelo.predict(devX)
    probas = modelo.predict_proba(devX)

    predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=devX.index, columns=cols)

    # Build scored dataset
    results_dev = devX.join(predictions, how='left')
    results_dev = results_dev.join(probabilities, how='left')
    results_dev = results_dev.join(dev['__target__'], how='left')
    results_dev = results_dev.rename(columns={'__target__': 'TARGET'})

    i = 0
    for real, pred in zip(devY, predictions):
        print(real, pred)
        i += 1
        if i > 5:
            break

    print(f1_score(devY, predictions, average=None))
    print(classification_report(devY, predictions))
    print(confusion_matrix(devY, predictions, labels=[1, 0]))


#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main_knn():
    mejor_metrica_valor=0
    # Entrada principal del programa
    print("---- Iniciando KNN...")

    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT, encoding="utf-8")
    print('-- Las 10 primeras líneas del csv son: --')
    print(ml_dataset.head(10))
    print("---- Estandarizamos en Unicode pasamos de atributos categoricos a numericos")
    coerce_to_unicode(ml_dataset)

    print("---- Tratamos el TARGET: " + TARGET)
    # Creamos la columna __target__ con el atributo a predecir
    target_map = {'0': 0, '1': 1}
    ml_dataset['__target__'] = ml_dataset[TARGET].map(str).map(target_map)
    ml_dataset['__target__'] = ml_dataset[TARGET]
    del ml_dataset[TARGET]

    print("---- Pasamos de atributos categoricos a numericos")
    # Borramos aquellas entradas de datos en las que el target sea null
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    numerical_features, text_features = separar_columnas(ml_dataset)
    print('-- Separamos atributos de texto de atributos categóricos --')
    text_features, categorical_features = separar_texto_categorico(ml_dataset, text_features)
    print('-- Separamos atributos numéricos de atributos categóricos -- \n\n')
    numerical_features, categorical_features = separar_numerico_categorico(ml_dataset, numerical_features,
                                                                           categorical_features)

    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)


    print("---- Dataset empleado")
    print(ml_dataset.head(5))  # Imprimimos las primeras 5 lineas

    print("---- Preprocesamos los datos")

    # preprocesar_datos(dataset, categorical_features, text_features, numerical_features, drop_rows_when_missing, impute_when_missing, rescale_features)
    train, dev = preprocesar_datos(ml_dataset, numerical_features, categorical_features, text_features)

    print("---- Dataset preprocesado")
    print("TRAIN: ")
    print(train.head(5))  # Imprimimos las primeras 5 lineas
    print(train['__target__'].value_counts())
    print("DEV: ")
    print(dev.head(5))
    print(dev['__target__'].value_counts())

    trainX = train.drop('__target__', axis=1)
    # trainY = train['__target__']
    devX = dev.drop('__target__', axis=1)
    # devY = devt['__target__']
    trainY = np.array(train['__target__'])
    devY = np.array(dev['__target__'])
    trainX.to_csv('trainX.csv', index=False)
    print("---- Undersampling... o Oversampling??")
    # Consultar el siguiente tutorial https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
    # Hacer un undersample para que las clases menos representadas estén más presentes para el entreno
    # sampling_strategy = diccionario {'clase': valor} -> multiclass
    # buscar informacion sobre RandomUnderSampler XXX =RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    # trainX,trainY = XXXX.fit_resample(trainX,trainY)
    # devX,devY = XXX.fit_resample(devX, devY)
    # pensar si lo que quiero hacer es oversampling

    print("---- Iniciando barrido de parámetros ")
    print("TRAINX: ")
    print(trainX.head(5))  # Imprimimos las primeras 5 lineas
    print("DEVX: ")
    print(devX.head(5))
    print("##########################")
    imprimir_faltantes(trainX)

    resultados_modelos = []
    mejor_f1 = 0
    mejores_parametros = (None, None, None)

    for k in range(K_MIN, K_MAX + 1):
        for p in range(P_MIN, P_MAX + 1):
            for w in D:
                print(f"KNN -> k: {k} p:{p} w:{w}")
                # weights : {'uniform', 'distance'} or callable, default='uniform'
                # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
                clf = KNeighborsClassifier(n_neighbors=k,
                                           weights=w,
                                           algorithm='auto',
                                           leaf_size=30,
                                           p=p)
                # Indica que las clases están balanceadas -> paso anterior de undersampling
                clf.class_weight = "balanced"
                # Ajustamos el modelo a nuestro train
                clf.fit(trainX, trainY)
                # Aplicamos el dev para ver el comportamiento del modelo
                # comprobar_modelo(clf, devX, devY, target_map)

                predictions = clf.predict(devX)
                probas = clf.predict_proba(devX)

                predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
                print("##########################target_map")
                print(target_map)
                cols = [f'probability_of_value_{target_map.get(label, label)}' for label in clf.classes_]

                print("######################### PROBBBASSS")
                print(probas)
                print("######################### DEVX.INDEX")
                print(devX.index)
                print(devX.head(5))
                print("######################### COLS")
                print(cols)
                print("Forma de probas:", probas.shape)
                print("Longitud de cols:", len(cols))
                probabilities = pd.DataFrame(data=probas, index=devX.index, columns=cols)

                results_dev = devX.join(predictions, how='left')
                results_dev = results_dev.join(probabilities, how='left')
                results_dev = results_dev.join(dev['__target__'], how='left')
                results_dev = results_dev.rename(columns={'__target__': 'TARGET'})

                i = 0
                for real, pred in zip(devY, predictions):
                    print(real, pred)
                    i += 1
                    if i > 5:
                        break

                print(f1_score(devY, predictions, average=None))
                print(classification_report(devY, predictions))
                print(confusion_matrix(devY, predictions))

                metrica_valor = calcular_metrica(devY, predictions, metrica_seleccionada)
                if metrica_valor > mejor_metrica_valor:
                    mejor_metrica_valor = metrica_valor
                    mejores_parametros = (k, p, w )

                resultados_modelos.append([k, p, w, metrica_valor])

    with open('modelos_knn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'p', 'w', metrica_seleccionada])
        writer.writerows(resultados_modelos)

    clf = KNeighborsClassifier(n_neighbors=mejores_parametros[0],
                               weights=mejores_parametros[2],
                               algorithm='auto',
                               leaf_size=30,
                               p=mejores_parametros[1])

    clf.fit(trainX, trainY)

    nombreModel = "mejor_modelo_knn.sav"  # Nombre del archivo donde guardarás el modelo

    saved_model = pickle.dump(clf, open(nombreModel, 'wb'))



    print('Mejor modelo de KNN guardado correctamente empleando Pickle')
    with open('mejor_modelo_knn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'p', 'w', 'Mejor ' + metrica_seleccionada])
        writer.writerow([mejores_parametros[0], mejores_parametros[1], mejores_parametros[2], mejor_metrica_valor])


def main_randomForest():
    mejor_metrica_valor = 0
    # Entrada principal del programa
    print("---- Iniciando Random Forest...")

    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT, encoding="utf-8")
    print('-- Las 10 primeras líneas del csv son: --')
    print(ml_dataset.head(10))

    print("---- Estandarizamos en Unicode pasamos de atributos categoricos a numericos")
    coerce_to_unicode(ml_dataset)

    print("---- Tratamos el TARGET: " + TARGET)
    # Creamos la columna __target__ con el atributo a predecir
    target_map = {'0': 0, '1': 1}
    ml_dataset['__target__'] = ml_dataset[TARGET].map(str).map(target_map)
    ml_dataset['__target__'] = ml_dataset[TARGET]
    del ml_dataset[TARGET]

    print("---- Pasamos de atributos categoricos a numericos")
    # Borramos aquellas entradas de datos en las que el target sea null
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    numerical_features, text_features = separar_columnas(ml_dataset)
    print('-- Separamos atributos de texto de atributos categóricos --')
    text_features, categorical_features = separar_texto_categorico(ml_dataset, text_features)
    print('-- Separamos atributos numéricos de atributos categóricos -- \n\n')
    numerical_features, categorical_features = separar_numerico_categorico(ml_dataset, numerical_features,
                                                                           categorical_features)

    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)

    print("---- Dataset empleado")
    print(ml_dataset.head(5))  # Imprimimos las primeras 5 lineas

    print("---- Preprocesamos los datos")
    # preprocesar_datos(dataset, categorical_features, text_features, numerical_features, drop_rows_when_missing, impute_when_missing, rescale_features)
    train, dev = preprocesar_datos(ml_dataset, numerical_features, categorical_features, text_features)

    trainX = train.drop('__target__', axis=1)
    # trainY = train['__target__']
    devX = dev.drop('__target__', axis=1)
    # devY = devt['__target__']
    trainY = np.array(train['__target__'])
    devY = np.array(dev['__target__'])

    print("---- Iniciando barrido de parámetros ")
    print("TRAINX: ")
    print(trainX.head(5))  # Imprimimos las primeras 5 lineas
    print("DEVX: ")
    print(devX.head(5))



    resultados_modelos = []
    mejor_f1 = 0
    mejores_parametros = (None,None, None, None)

    # Definir rangos para los hiperparámetros del Random Forest
    N_ESTIMATORS_RANGE = range(MINESTIMATOR, MAXESTIMATOR)  # Número de árboles
    MAX_DEPTH_RANGE = range(MINDEPTH, MAXDEPTH)  # Profundidades del 1 al 10
    MIN_SAMPLES_SPLIT_RANGE = range(MINSAMPLE, MAXSAMPLE)  # Mínimo número de muestras requeridas para dividir un nodo
    CRITERIA = ['gini', 'entropy']  # Criterios para medir la calidad de una división
    for n_estimators in N_ESTIMATORS_RANGE:
        for depth in MAX_DEPTH_RANGE:
            for min_samples in MIN_SAMPLES_SPLIT_RANGE:
                for criterion in CRITERIA:
                    clf = RandomForestClassifier(n_estimators=n_estimators,
                                                    max_depth=depth,
                                                    min_samples_split=min_samples,
                                                    criterion=criterion,
                                                    class_weight="balanced")

                    # Indica que las clases están balanceadas -> paso anterior de undersampling
                    clf.class_weight = "balanced"
                    # Ajustamos el modelo a nuestro train
                    clf.fit(trainX, trainY)

                    # Aplicamos el dev para ver el comportamiento del modelo
                    # comprobar_modelo(clf, devX, devY, target_map)

                    predictions = clf.predict(devX)
                    probas = clf.predict_proba(devX)

                    predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
                    print("##########################target_map")
                    print(target_map)
                    cols = [f'probability_of_value_{target_map.get(label, label)}' for label in clf.classes_]

                    print("######################### PROBBBASSS")
                    print(probas)
                    print("######################### DEVX.INDEX")
                    print(devX.index)
                    print(devX.head(5))
                    print("######################### COLS")
                    print(cols)
                    print("Forma de probas:", probas.shape)
                    print("Longitud de cols:", len(cols))
                    probabilities = pd.DataFrame(data=probas, index=devX.index, columns=cols)



                    results_dev = devX.join(predictions, how='left')
                    results_dev = results_dev.join(probabilities, how='left')
                    results_dev = results_dev.join(dev['__target__'], how='left')
                    results_dev = results_dev.rename(columns={'__target__': 'TARGET'})

                    i = 0
                    for real, pred in zip(devY, predictions):
                        print(real, pred)
                        i += 1
                        if i > 5:
                            break

                    print(f1_score(devY, predictions, average=None))
                    print(classification_report(devY, predictions))
                    print(confusion_matrix(devY, predictions))

                    metrica_valor = calcular_metrica(devY, predictions, metrica_seleccionada)

                    resultados_modelos.append([n_estimators, depth, min_samples, criterion, metrica_valor])
                    if metrica_valor > mejor_metrica_valor:
                        mejor_metrica_valor = metrica_valor
                        mejores_parametros = (n_estimators, depth, min_samples, criterion)
                    # Guardar resultados en la lista


    with open('modelos_random_forest.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Num estimadores','Profundidad', 'Mínimo Muestras', 'Criterio', metrica_seleccionada])
        writer.writerows(resultados_modelos)

    nombreModel = "mejor_modelo_random_forest.sav"  # Nombre del archivo donde guardarás el modelo

    # Guardar el mejor modelo a disco
    with open(nombreModel, 'wb') as file:
        pickle.dump(clf, file)

    with open('mejor_modelo_random_forest.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Num estimadores','Profundidad', 'Mínimo Muestras', 'Criterio', metrica_seleccionada])
        writer.writerow([mejores_parametros[0], mejores_parametros[1], mejores_parametros[2],mejores_parametros[3], mejor_metrica_valor])

    # Supongamos que la escritura a CSV se realiza aquí

    # Este bloque de código se centra en la lógica de adaptación al Random Forest, sin ejecutar operaciones de E/S o ajuste de modelo real.
    # Los comentarios indican dónde el código real realizaría estas acciones.


def main_tree():
    mejor_metrica_valor = 0
    # Entrada principal del programa
    print("---- Iniciando Decision Tree...")

    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT, encoding="utf-8")
    print('-- Las 10 primeras líneas del csv son: --')
    print(ml_dataset.head(10))

    print("---- Estandarizamos en Unicode pasamos de atributos categoricos a numericos")
    coerce_to_unicode(ml_dataset)

    print("---- Tratamos el TARGET: " + TARGET)
    # Creamos la columna __target__ con el atributo a predecir
    target_map = {'0': 0, '1': 1}
    ml_dataset['__target__'] = ml_dataset[TARGET].map(str).map(target_map)
    ml_dataset['__target__'] = ml_dataset[TARGET]
    del ml_dataset[TARGET]

    print("---- Pasamos de atributos categoricos a numericos")
    # Borramos aquellas entradas de datos en las que el target sea null
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    numerical_features, text_features = separar_columnas(ml_dataset)
    print('-- Separamos atributos de texto de atributos categóricos --')
    text_features, categorical_features = separar_texto_categorico(ml_dataset, text_features)
    print('-- Separamos atributos numéricos de atributos categóricos -- \n\n')
    numerical_features, categorical_features = separar_numerico_categorico(ml_dataset, numerical_features,
                                                                           categorical_features)

    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)

    print("---- Dataset empleado")
    print(ml_dataset.head(5))  # Imprimimos las primeras 5 lineas

    print("---- Preprocesamos los datos")
    # preprocesar_datos(dataset, categorical_features, text_features, numerical_features, drop_rows_when_missing, impute_when_missing, rescale_features)
    train, dev = preprocesar_datos(ml_dataset, numerical_features, categorical_features, text_features)
    trainX = train.drop('__target__', axis=1)
    trainY = np.array(train['__target__'])
    devX = dev.drop('__target__', axis=1)
    devY = np.array(dev['__target__'])

    # Iniciar barrido de parámetros para DecisionTree
    resultados_modelos = []
    mejor_f1 = 0
    mejores_parametros = (None, None, None)

    # Definir rangos para los hiperparámetros del árbol de decisión
    MAX_DEPTH_RANGE = range(MINDEPTH, MAXDEPTH)
    MIN_SAMPLES_SPLIT_RANGE = range(MINSAMPLE, MAXSAMPLE)
    CRITERIA = ['gini', 'entropy']  # Criterios para medir la calidad de una división

    for depth in MAX_DEPTH_RANGE:
        for min_samples in MIN_SAMPLES_SPLIT_RANGE:
            for criterion in CRITERIA:
                clf = DecisionTreeClassifier(max_depth=depth,
                                             min_samples_split=min_samples,
                                             criterion=criterion,
                                             class_weight="balanced")

                clf.fit(trainX, trainY)
                predictions = clf.predict(devX)

                metrica_valor = calcular_metrica(devY, predictions, metrica_seleccionada)
                if metrica_valor > mejor_metrica_valor:
                    mejor_metrica_valor = metrica_valor
                    mejores_parametros = (depth, min_samples, criterion)

                resultados_modelos.append([depth, min_samples, criterion, metrica_valor])

                print(f1_score(devY, predictions, average=None))
                print(classification_report(devY, predictions))
                print(confusion_matrix(devY, predictions))

    # Guardar resultados y mejor modelo en archivos CSV
    with open('modelos_decision_tree.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Profundidad', 'Mínimo Muestras', 'Criterio', metrica_seleccionada])
        writer.writerows(resultados_modelos)


    clf= DecisionTreeClassifier(max_depth=mejores_parametros[0], min_samples_split=mejores_parametros[1], criterion=mejores_parametros[2])
    clf.fit(trainX, trainY)

    nombreModel = "mejor_modelo_decision_tree.sav"  # Nombre del archivo donde guardarás el modelo


    saved_model  = pickle.dump(clf, open(nombreModel,'wb'))


    #with open(nombreModel, 'wb') as file:
    #    pickle.dump(clf, file)

    with open('mejor_modelo_decision_tree.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Profundidad', 'Mínimo Muestras', 'Criterio', metrica_seleccionada])
        writer.writerow([mejores_parametros[0], mejores_parametros[1], mejores_parametros[2], mejor_metrica_valor])




if __name__ == "__main__":
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt(argv[1:],
                                    'hi:o:m:',
                                    ['help', 'input=', 'output=', 'method=', 'remove-missing=',
                                     'sampling=', 'scaling=', 'sample-rate=', 'clean-text=',
                                     'categorical-numerical=', 'distance=', 'k-min=', 'k-max=',
                                     'p-min=', 'p-max=', 'min-depth=', 'max-depth=', 'min-sample=',
                                     'max-sample=', 'min-estimator=', 'max-estimator=', 'seed=','dev-size=','metric=','target='])


    except getopt.GetoptError as err:
        # Error al parsear las opciones del comando
        print("ERROR: ", err)
        exit(1)

    # Registramos la configuración del script
    load_options(options)
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    if (METODO == "knn"):
        main_knn()
    elif (METODO == "randomForest"):
        main_randomForest()
    elif (METODO == "decisionTree"):
        main_tree()
    elif (METODO == 45):
        print("Debes introducir un método a realizar con -m")
    else:
        print("ERROR: Solo valen las opciones knn, randomForest, forest")
