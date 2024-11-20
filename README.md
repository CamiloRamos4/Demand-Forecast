# Demand-Forecast Ejemplo 1

    Code 1:

    import pandas as pd

    import numpy as np

    from prophet import Prophet

    import matplotlib.pyplot as plt

    #Crear datos sintéticos

    np.random.seed(42)

    dates = pd.date_range(start='2024-10-01', end='2024-10-31')

    ventas = np.random.randint(400, 600, size=len(dates))

    #Añadir impacto del clima y eventos

    clima = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])

    evento = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])

    #Aumentar ventas en días con clima favorable y eventos

    ventas = ventas + clima * 200 + evento * 300

    #Crear DataFrame

    data = {
        'Fecha': dates,
        'Ventas': ventas,
        'Clima': clima,
        'Evento': evento
    }

    df = pd.DataFrame(data)

    df['Fecha'] = pd.to_datetime(df['Fecha'])

    #Renombrar columnas para Prophet

    df.rename(columns={'Fecha': 'ds', 'Ventas': 'y'}, inplace=True)

    #Mostrar los primeros registros del DataFrame

    print(df.head())



# Demand-Forecast Ejemplo 2

    pip install prophet

    import pandas as pd

    import numpy as np

    #Crear datos históricos simulados

    np.random.seed(42)  # Fijar semilla para reproducibilidad

    dates = pd.date_range(start="2023-01-01", end="2023-12-31")

    sales = np.random.randint(50, 150, size=len(dates))  # Ventas aleatorias

    temperature = np.random.randint(10, 35, size=len(dates))  # Temperatura diaria

    holiday = [1 if date.month == 12 and date.day in [24, 25, 31] else 0 for date in dates]  # Feriados en diciembre

    #Crear DataFrame

    data = pd.DataFrame({
        "ds": dates,
        "y": sales,
        "temperature": temperature,
        "holiday": holiday
    })

    #Mostrar datos simulados

    print(data.head())

    from prophet import Prophet

    #Crear el modelo Prophet

    model = Prophet()

    #Añadir regresores adicionales

    model.add_regressor('temperature')

    model.add_regressor('holiday')

    #Entrenar el modelo

    model.fit(data)

    #Crear un marco de tiempo para 30 días futuros

    future = model.make_future_dataframe(periods=30)

    #Generar predicciones

    forecast = model.predict(future)

    #Mostrar resultados

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  # Predicción y rangos

    #Añadir los valores de los regresores para las fechas futuras

    #(puedes ajustar según tus datos reales)

    future['temperature'] = np.random.randint(10, 35, size=len(future))  # Simulación de temperatura

    future['holiday'] = [1 if date.month == 12 and date.day in [24, 25, 31] else 0 for date in future['ds']]  # Feriados

    import matplotlib.pyplot as plt

    #Graficar predicciones

    model.plot(forecast)

    plt.title("Predicción de Ventas para la Panadería")

    plt.xlabel("Fecha")

    plt.ylabel("Ventas")

    plt.show()

    model.plot_components(forecast)

    plt.show()

    import pandas as pd

    import numpy as np

    from prophet import Prophet

    import matplotlib.pyplot as plt

    #Crear datos históricos simulados

    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", end="2023-12-31")

    sales = np.random.randint(50, 150, size=len(dates))
    temperature = np.random.randint(10, 35, size=len(dates))
    holiday = [1 if date.month == 12 and date.day in [24, 25, 31] else 0 for date in dates]

    #Crear DataFrame
    data = pd.DataFrame({
        "ds": dates,
        "y": sales,
        "temperature": temperature,
        "holiday": holiday
    })

    #Crear y configurar el modelo Prophet

    model = Prophet()

    model.add_regressor('temperature')

    model.add_regressor('holiday')

    #Entrenar el modelo

    model.fit(data)

    #Crear marco de tiempo para predicciones futuras

    future = model.make_future_dataframe(periods=30)

    future['temperature'] = np.random.randint(10, 35, size=len(future))

    future['holiday'] = [1 if date.month == 12 and date.day in [24, 25, 31] else 0 for date in future['ds']]

    #Generar predicciones

    forecast = model.predict(future)

    #Graficar predicciones

    model.plot(forecast)

    plt.title("Predicción de Ventas para la Panadería")

    plt.xlabel("Fecha")

    plt.ylabel("Ventas")

    plt.show()

    #Graficar componentes del modelo

    model.plot_components(forecast)

    plt.show()

    #Generar una conclusión basada en los resultados

    def generar_conclusion(data, forecast):
        # Extraer componentes principales
        tendencia = forecast['trend']
        predicciones = forecast['yhat']

        # Promedio y tendencia
        promedio_actual = data['y'].mean()
        promedio_predicho = predicciones.tail(30).mean()
        diferencia = promedio_predicho - promedio_actual

    # Impacto de regresores
    impacto_regresores = forecast['additive_terms'].mean()  # Incluye regresores como clima y feriados

    conclusion = f"""
    Conclusión del Modelo:
    1. Ventas promedio históricas: {promedio_actual:.2f}.
    2. Ventas promedio predichas para los próximos 30 días: {promedio_predicho:.2f}.
       {'Se espera un incremento' if diferencia > 0 else 'Se espera una disminución'} de {abs(diferencia):.2f} unidades diarias en promedio.
    3. Impacto promedio de regresores:
       - Contribución estimada promedio de {impacto_regresores:.2f} unidades debido a factores como clima y feriados.
    4. El modelo indica que la tendencia general de ventas es {'ascendente' if tendencia.iloc[-1] > tendencia.iloc[0] else 'descendente'}.
    """
    return conclusion

    # Imprimir la conclusión
    conclusion = generar_conclusion(data, forecast)
    print(conclusion)
