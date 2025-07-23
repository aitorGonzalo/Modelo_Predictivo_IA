import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def graficar(archivo):
    try:
        data = pd.read_csv(archivo)
        fig, ax1 = plt.subplots()
        secondary_axis = False

        if 'Perplexity' in data.columns and 'Num_Topics' in data.columns:
            color = 'tab:red'
            ax1.set_xlabel('Número de Temas')
            ax1.set_ylabel('Perplexity', color=color)
            ax1.plot(data["Num_Topics"], data["Perplexity"], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.invert_yaxis()
            secondary_axis = True
        elif 'Coherence' in data.columns and 'Num_Topics' in data.columns:
            ax1.set_xlabel('Número de Temas')
            ax1.set_ylabel('Coherence', color='tab:blue')
            ax1.plot(data["Num_Topics"], data["Coherence"], color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

        if secondary_axis and 'Coherence' in data.columns:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Coherence', color=color)
            ax2.plot(data["Num_Topics"], data["Coherence"], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Puntuaciones de Perplexity y Coherence por Número de Temas')
        plt.get_current_fig_manager().set_window_title(archivo)

    except KeyError as e:
        print(f"Error al procesar el archivo {archivo}: columna {e} no encontrada.")
    except Exception as e:
        print(f"Error al procesar el archivo {archivo}: {e}")

def main(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                graficar(filepath)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grafica archivos CSV en un directorio especificado.")
    parser.add_argument("directory", type=str, help="El directorio que contiene los archivos CSV.")
    args = parser.parse_args()
    main(args.directory)
