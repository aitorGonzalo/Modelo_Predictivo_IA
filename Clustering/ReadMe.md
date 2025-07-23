# Instrucciones para la Ejecución del Programa de Topic Modelling

## Preparación

Para ejecutar este programa, muy importante,debes instalar las dependencias necesarias en un entorno Python 3.11 :

```bash
pip install -r requirements.txt
```

## Estructura del Programa

El programa consta de dos partes principales:

- `TopicModelling.py`: Script principal de Python.
- `config.json`: Archivo de configuración necesario para ejecutar el script.

## Ejecución

Para llamar al programa, utiliza el siguiente comando en la terminal:

```bash
python TopicModelling.py [ruta a config.json]
```

## Salida del Programa

Al finalizar la ejecución, el programa generará un archivo CSV con la información de la ejecución. Aunque se graficarán los resultados, como los mejores modelos se basan únicamente en un número específico de tópicos, no se mostrará nada en el gráfico. Para verificar la ejecución, será necesario revisar el archivo CSV, que llevará un nombre descriptivo.

Los tópicos se guardaran en un topics txt.
## Recomendaciones
Si se quiere crear una visualizacion, lo mejor es que kmin=kmax para que asi el topics.txt y la visualizacion sea de lo mismo ya que solamente se guarda del ultimo modelo ejecutado




