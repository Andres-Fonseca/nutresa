# Sistema de Recomendación de Productos

Este proyecto desarrolla un modelo de recomendación semanal para anticipar las necesidades de compra de los clientes, basado en sus historiales de transacciones. Esto permite al equipo de ventas priorizar productos con alta probabilidad de compra y mejorar la eficiencia en las interacciones con los clientes.

## Descripción General

El objetivo es implementar un modelo de machine learning que prediga la probabilidad de compra de productos específicos en un período determinado para cada cliente. Esta recomendación semanal genera una "Orden Sugerida" para ayudar al equipo de ventas a preparar visitas, maximizando la eficiencia al anticipar los productos que un cliente es más probable que adquiera.

Este sistema está diseñado para un grupo de ventas que realiza visitas tanto directas como indirectas, y busca enfocarse en canales, categorías, y tipologías de clientes con menor frecuencia de compra para maximizar el impacto.

## Características

- **Generación de recomendaciones de productos**: Predicción de la probabilidad de compra para productos específicos en la semana próxima.
- **Modelo de predicción personalizada**: Ajustado a la historia y patrones de compra de cada cliente.
- **Estrategias de segmentación**: Prioriza clientes de canales, sectores, marcas y categorías con menor frecuencia de compra.
- **Optimización de las interacciones de ventas**: Reduce el tiempo invertido en productos de menor interés y mejora la preparación del equipo de ventas.

## Requisitos Previos

### Tecnologías
- Python 3.8 o superior

### Librerías y dependencias
- `pandas`: Manipulación de datos
- `numpy`: Operaciones numéricas
- `scikit-learn`: Modelado de machine learning
- `xgboost`, `lightgbm`: Modelos avanzados de boosting
- `matplotlib`, `seaborn`: Visualización de datos
- `sqlite3`: Base de datos para almacenamiento de historial


├── data/                    # Carpeta para los archivos de datos (e.g., historial de ventas, clientes, productos)
├── notebooks/               # Notebooks de Jupyter para pruebas y experimentación
├── src/                     # Código fuente del modelo y procesamiento de datos
│   ├── data_processing.py   # Procesamiento y limpieza de datos
│   ├── feature_engineering.py # Ingeniería de características fechas
│   ├── model_training.py    # Entrenamiento del modelo 
│   ├── recommendations.py   # Generación de recomendaciones
