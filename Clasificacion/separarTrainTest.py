import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV
df = pd.read_csv('Modified_AirlinesReviews.csv')

# Dividir el DataFrame
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Escribir los conjuntos de entrenamiento y prueba en archivos CSV
train.to_csv('Train.csv', index=False)
test.to_csv('Test.csv', index=False)