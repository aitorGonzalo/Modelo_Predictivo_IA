import json
import pandas as pd
import nltk
from gensim.models import Phrases, Nmf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import gensim
import matplotlib.pyplot as plt
import argparse
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.matutils import corpus2csc


#################################################################################
###                 DEFINICIONES DE FUNCIONES                               #####
#################################################################################
def load_options(json_path):
    """ Cargar configuración desde un archivo JSON y actualizar variables globales. """
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Actualiza las variables globales con las configuraciones del JSON
    global INPUT, TARGET, TARGET2, NOSOTROS, COMPETIDOR, NEUTRO, UNIGRAM, BIGRAM, TRIGRAM, METODO, KMIN, KMAX, INTERVALO, PASSES, PALABRAS, VISUALIZAR, MALONOSOTROS, BUENONOSOTROS, MALOCOMPETIDOR, BUENOCOMPETIDOR,COHERENCE
    INPUT = config.get("INPUT")
    TARGET = config.get("TARGET")
    TARGET2 = config.get("TARGET2")
    NOSOTROS = config.get("NOSOTROS")
    COMPETIDOR = config.get("COMPETIDOR")
    NEUTRO = config.get("NEUTRO")
    UNIGRAM = config.get("UNIGRAM")
    BIGRAM = config.get("BIGRAM")
    TRIGRAM = config.get("TRIGRAM")
    METODO = config.get("METODO")
    KMIN = config.get("KMIN")
    KMAX = config.get("KMAX")
    INTERVALO = config.get("INTERVALO")
    PASSES = config.get("PASSES")
    PALABRAS = config.get("PALABRAS")
    VISUALIZAR = config.get("VISUALIZAR")
    MALONOSOTROS = config.get("MALONOSOTROS")
    BUENONOSOTROS = config.get("BUENONOSOTROS")
    MALOCOMPETIDOR = config.get("MALOCOMPETIDOR")
    BUENOCOMPETIDOR = config.get("BUENOCOMPETIDOR")
    COHERENCE = config.get("COHERENCE")   #c_v / u_mass / c_uci / c_npmi


def print_config():
    """ Imprime todas las configuraciones cargadas desde el archivo JSON. """
    print("Configuración actual:")
    print(f"INPUT: {INPUT}")
    print(f"TARGET: {TARGET}")
    print(f"TARGET2: {TARGET2}")
    print(f"NOSOTROS: {NOSOTROS}")
    print(f"COMPETIDOR: {COMPETIDOR}")
    print(f"NEUTRO: {NEUTRO}")
    print(f"UNIGRAM: {UNIGRAM}")
    print(f"BIGRAM: {BIGRAM}")
    print(f"TRIGRAM: {TRIGRAM}")
    print(f"METODO: {METODO}")
    print(f"KMIN: {KMIN}")
    print(f"KMAX: {KMAX}")
    print(f"intervalo: {INTERVALO}")
    print(f"passes: {PASSES}")
    print(f"n palabras impresas: {PALABRAS}")
    print(f"visualizar (guarda un html): {VISUALIZAR}")
    print(f"coherence: {COHERENCE}")


def cargarJson():
    parser = argparse.ArgumentParser(description='Analiza datos de reseñas de aerolíneas.')
    parser.add_argument('config_path', type=str, help='Ruta al archivo JSON de configuración.')
    args = parser.parse_args()

    # Cargar configuraciones desde el JSON especificado
    load_options(args.config_path)


def graficar(archivo):
    data = pd.read_csv(archivo)

    # Creando el gráfico
    fig, ax1 = plt.subplots()

    if 'Perplexity' in data.columns and METODO != 'NMF':
        color = 'tab:red'
        ax1.set_xlabel('Número de Temas')
        ax1.set_ylabel('Perplexity', color=color)
        ax1.plot(data["Num_Topics"], data["Perplexity"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        secondary_axis = True
    else:
        ax1.set_xlabel('Número de Temas')
        ax1.set_ylabel('Coherence', color='tab:blue')
        ax1.plot(data["Num_Topics"], data["Coherence"], color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        secondary_axis = False

    if secondary_axis:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Coherence', color=color)
        ax2.plot(data["Num_Topics"], data["Coherence"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Puntuaciones de Perplexity y Coherence por Número de Temas')
    plt.get_current_fig_manager().set_window_title(archivo)


def separarCSV(ml_dataset, nosotros, competidor):
    # Filtrar los datos para la aerolínea en la variable 'nosotros'
    CSVNosotros = ml_dataset[ml_dataset['Airline'] == nosotros]

    # Filtrar los datos para la aerolínea en la variable 'competidor'
    CSVCompetidor = ml_dataset[ml_dataset['Airline'] == competidor]

    print("CSV NOSOTROS")
    print(CSVNosotros)
    print("CSV COMPETIDOR")
    print(CSVCompetidor)

    return CSVNosotros, CSVCompetidor


def separarClasificaciones(df):
    df_malo = df[df['Overall Rating'].between(0, 4, inclusive='both')]
    df_neutro = df[df['Overall Rating'].between(5, 7, inclusive='both')]
    df_bueno = df[df['Overall Rating'].between(8, 10, inclusive='both')]

    return df_malo, df_neutro, df_bueno


def normalizar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto_letras = ''.join(c for c in texto if c.isalpha() or c.isspace())  # Conservar solo letras y espacios
    palabras = word_tokenize(texto_letras)  # Tokenizar
    stop_words = set(stopwords.words('english'))  # Eliminar stopwords
    palabras = [p for p in palabras if p.lower() not in stop_words]

    return palabras


def aplicar_lda(cuales, text_data, num_topics, results_df, PASSES=13):
    processed_text_data = []

    # Generar bigramas si BIGRAM es True o TRIGRAM es True porque los trigramas dependen de los bigramas
    if BIGRAM or TRIGRAM:
        bigram = Phrases(text_data, min_count=5, threshold=10)  # Ajusta estos valores según tus necesidades
        bigram_mod = Phraser(bigram)
        if TRIGRAM:
            trigram = Phrases(bigram_mod[text_data], min_count=1, threshold=1)
            trigram_mod = Phraser(trigram)

    for text in text_data:
        if TRIGRAM:
            processed_text = trigram_mod[bigram_mod[text]]
        elif BIGRAM:
            processed_text = bigram_mod[text]
        else:
            processed_text = text

        # Filtrar por n-grams específicos si es necesario
        if not UNIGRAM:
            processed_text = [word for word in processed_text if '_' in word]  # Retener solo n-grams
        if not BIGRAM:
            processed_text = [word for word in processed_text if word.count('_') != 1]
        if not TRIGRAM:
            processed_text = [word for word in processed_text if word.count('_') != 2]

        processed_text_data.append(processed_text)

    # Crear un diccionario y un corpus usando los textos procesados
    dictionary = corpora.Dictionary(processed_text_data)
    corpus = [dictionary.doc2bow(text) for text in processed_text_data]

    # Entrenar el modelo LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=PASSES,
                                                alpha='auto', eta='auto')
    topics = lda_model.print_topics(num_words=PALABRAS)

    with open('topics.txt', 'w') as file:
        for num, topic in topics:
            # Escribir el número de tópico y las palabras clave asociadas
            file.write(f"Topic {num}: {topic}\n")

    # Calcular y mostrar la perplejidad y la coherencia
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_text_data, dictionary=dictionary,
                                        coherence=COHERENCE)
    coherence_lda = coherence_model_lda.get_coherence()

    print("Perplexity: ", perplexity)
    print('Coherencia del modelo LDA:', coherence_lda)


    # Registrar los resultados
    results_df.loc[len(results_df)] = [num_topics, perplexity, coherence_lda]
    if VISUALIZAR:
        visualizacion = gensimvis.prepare( topic_model= lda_model,corpus=corpus , dictionary=dictionary)
        pyLDAvis.save_html(visualizacion,
                           f'visualizacion_{cuales}_{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.html')

    return results_df


def aplicar_nmf(text_data, num_topics, results_df):
    processed_text_data = []

    # Preparación de modelos de Phraser
    bigram = Phrases(text_data, min_count=5, threshold=10)
    bigram_mod = Phraser(bigram)

    trigram_mod = None
    if TRIGRAM:
        temp_data = [bigram_mod[doc] for doc in text_data]
        trigram = Phrases(temp_data, min_count=1, threshold=1)
        trigram_mod = Phraser(trigram)

    # Procesar texto para aplicar modelos n-gram
    for text in text_data:
        if TRIGRAM:
            processed_text = trigram_mod[bigram_mod[text]]
        elif BIGRAM:
            processed_text = bigram_mod[text]
        else:
            processed_text = text

        # Filtrado de n-grams según sea necesario
        if not UNIGRAM:
            processed_text = [word for word in processed_text if '_' in word]
        if not BIGRAM:
            processed_text = [word for word in processed_text if word.count('_') != 1]
        if not TRIGRAM:
            processed_text = [word for word in processed_text if word.count('_') != 2]

        processed_text_data.append(processed_text)

    # Crear diccionario y corpus
    dictionary = corpora.Dictionary(processed_text_data)
    corpus = [dictionary.doc2bow(text) for text in processed_text_data]
    corpus_csc = corpus2csc(corpus, num_terms=len(dictionary))  # Convertir el corpus a CSC

    # Entrenar el modelo NMF
    nmf_model = Nmf(corpus_csc, num_topics=num_topics, id2word=dictionary, passes=PASSES)

    # Mostrar los temas
    topics = nmf_model.print_topics(num_words=PALABRAS)
    for topic in topics:
        print(topic)

    # Calcular la coherencia
    coherence_model_nmf = CoherenceModel(model=nmf_model, texts=processed_text_data, dictionary=dictionary,
                                         coherence='c_v')
    coherence_nmf = coherence_model_nmf.get_coherence()

    print('Coherencia del modelo NMF:', coherence_nmf)

    # Registrar los resultados
    new_row = pd.DataFrame({'Num_Topics': [num_topics], 'Coherence': [coherence_nmf]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df


#################################################################################
###                        PROGRAMA PRINCIPAL                               #####
#################################################################################
def main():
    ml_dataset = pd.read_csv(INPUT, encoding="utf-8")
    if TARGET2 is not None:
        # Unimos las columnas TARGET y TARGET2 en una sola y llenamos los valores nulos con una cadena vacía antes de unir
        ml_dataset['target'] = ml_dataset[TARGET].fillna('') + ml_dataset[TARGET2].fillna('')
        # Si necesitas que haya un espacio o algún delimitador entre los valores de ambas columnas, puedes modificar la línea anterior así:
        # ml_dataset['target'] = ml_dataset[TARGET].fillna('') + ' ' + ml_dataset[TARGET2].fillna('')

        # Eliminamos las columnas originales si ya no se necesitan
        ml_dataset.drop(columns=[TARGET, TARGET2], inplace=True)
    else:
        # Si TARGET2 es None, solo renombramos la columna TARGET a 'target'
        ml_dataset.rename(columns={TARGET: 'target'}, inplace=True)

    # Resultado final
    print(ml_dataset.head())

    # separar csv de competidor y nuestro
    csvnosotros, csvcompetidor = separarCSV(ml_dataset, NOSOTROS, COMPETIDOR)

    # separar csv dependiendo de las notas
    MaloNosotros, NeutroNosotros, BuenoNosotros = separarClasificaciones(csvnosotros)

    MaloComp, NeutroComp, BuenoComp = separarClasificaciones(csvcompetidor)
    # guardar en un array para que sea mas facil manejarlo
    if NEUTRO == 1:
        NuestroCSV = [MaloNosotros, NeutroNosotros, BuenoNosotros]
        CompetidorCSV = [MaloComp, NeutroComp, BuenoComp]
    else:
        NuestroCSV = [MaloNosotros, BuenoNosotros]
        CompetidorCSV = [MaloComp, BuenoComp]

    nltk.download('punkt')
    nltk.download('stopwords')

    # Preprocesar texto

    if MALONOSOTROS:

        if METODO == 'LDA':
            print(f"LDA aplicado en los nuestros CSV de K=" + str(KMIN) + "a K=" + str(KMAX))
            print("LDA en malo nosotros")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Perplexity': [],
                'Coherence': []
            })
        else:
            print(f"NMF aplicado en los nuestros CSV de K=" + str(KMIN) + "a K=" + str(KMAX))
            print("NMF en malo nosotros")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Coherence': []
            })

        for K in range(KMIN, KMAX + 1, INTERVALO):
            text_data = [normalizar_texto(doc) for doc in MaloNosotros['target'] if isinstance(doc, str)]
            # Aplicar LDA
            if text_data:
                if METODO == 'LDA':
                    results_df = aplicar_lda("MALONOSOTROS", text_data, K, results_df)
                    print(results_df)
                else:
                    results_df = aplicar_nmf(text_data, K, results_df)
                    print(results_df)

        filename = f"MALO_NOSOTROS____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv"
        results_df.to_csv(filename, index=False)

    if BUENONOSOTROS:

        if METODO == 'LDA':
            print("LDA en Bueno nosotros")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Perplexity': [],
                'Coherence': []
            })
        else:
            print("NMF en Bueno nosotros")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Coherence': []
            })

        for K in range(KMIN, KMAX + 1, INTERVALO):
            text_data = [normalizar_texto(doc) for doc in BuenoNosotros['target'] if isinstance(doc, str)]
            # Aplicar LDA
            if text_data:
                if METODO == 'LDA':
                    results_df = aplicar_lda("BUENONOSOTROS", text_data, K, results_df)
                    print(results_df)
                else:
                    results_df = aplicar_nmf(text_data, K, results_df)
                    print(results_df)

        filename = f"BUENO_NOSOTROS____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv"
        results_df.to_csv(filename, index=False)

    if MALOCOMPETIDOR:
        if METODO == 'LDA':
            print(f"LDA aplicado en los CSV de competidor de K=" + str(KMIN) + "a K=" + str(KMAX))
            print("LDA en malo competidor")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Perplexity': [],
                'Coherence': []
            })
        else:
            print(f"NMF aplicado en los CSV de competidor de K=" + str(KMIN) + "a K=" + str(KMAX))
            print("NMF en malo competidor")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Coherence': []
            })

    

        for K in range(KMIN, KMAX + 1, INTERVALO):
            text_data = [normalizar_texto(doc) for doc in MaloComp['target'] if isinstance(doc, str)]
            # Aplicar LDA
            if text_data:
                if METODO == 'LDA':
                    results_df = aplicar_lda("MALOCOMPETIDOR", text_data, K, results_df)
                    print(results_df)
                else:
                    results_df = aplicar_nmf(text_data, K, results_df)
                    print(results_df)

        filename = f"MALO_COMPETIDOR____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv"
        results_df.to_csv(filename, index=False)

    if BUENOCOMPETIDOR:

        if METODO == 'LDA':
            print("LDA en Bueno competidor")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Perplexity': [],
                'Coherence': []
            })
        else:
            print("NMF en Bueno competidor")
            results_df = pd.DataFrame({
                'Num_Topics': [],
                'Coherence': []
            })

        for K in range(KMIN, KMAX + 1, INTERVALO):
            text_data = [normalizar_texto(doc) for doc in BuenoComp['target'] if isinstance(doc, str)]
            # Aplicar LDA
            if text_data:
                if METODO == 'LDA':
                    results_df = aplicar_lda("BUENOCOMPETIDOR", text_data, K, results_df)
                    print(results_df)
                else:
                    results_df = aplicar_nmf(text_data, K, results_df)
                    print(results_df)

        filename = f"BUENO_COMPETIDOR____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv"
        results_df.to_csv(filename, index=False)


if __name__ == '__main__':
    cargarJson()
    print_config()

    main()

    if MALONOSOTROS:
        graficar(
            f'MALO_NOSOTROS____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv')
    if BUENONOSOTROS:
        graficar(
            f'BUENO_NOSOTROS____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv')
    if MALOCOMPETIDOR:
        graficar(
            f'MALO_COMPETIDOR____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv')
    if BUENOCOMPETIDOR:
        graficar(
            f'BUENO_COMPETIDOR____{METODO}_U={int(UNIGRAM)}_B={int(BIGRAM)}_T{int(TRIGRAM)}_KMIN={KMIN}_KMAX={KMAX}_INT={INTERVALO}_PASSES={PASSES}_COHERENCE={COHERENCE}.csv')
    plt.show()
