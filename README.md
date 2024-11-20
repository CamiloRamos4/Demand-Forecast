# Demand-Forecast Ejemplo 1

Code 1:
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Crear datos sintéticos
np.random.seed(42)
dates = pd.date_range(start='2024-10-01', end='2024-10-31')
ventas = np.random.randint(400, 600, size=len(dates))

# Añadir impacto del clima y eventos
clima = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
evento = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])

# Aumentar ventas en días con clima favorable y eventos
ventas = ventas + clima * 200 + evento * 300

# Crear DataFrame
data = {
    'Fecha': dates,
    'Ventas': ventas,
    'Clima': clima,
    'Evento': evento
}
df = pd.DataFrame(data)
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Renombrar columnas para Prophet
df.rename(columns={'Fecha': 'ds', 'Ventas': 'y'}, inplace=True)

# Mostrar los primeros registros del DataFrame
print(df.head())
