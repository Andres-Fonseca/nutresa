import pandas as pd
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import kstest
#\
import warnings
import inspect

import funciones_eda
from funciones_eda import *
    
    
#-------- datos generales--------------#    
def datos_generales(df):
    numero_de_datos=df.shape[0]
    numero_de_features=df.shape[1]
    print(f'El dataframe tiene {numero_de_datos} filas y {numero_de_features} columnas')


def borrar_caracteristicas(df, columnas):
    df=df.copy()
    for col in columnas:
        df.drop(col, axis=1, inplace=True)
    return df 

     
    
def categoricas(df):
    df_cat = df.select_dtypes(include = \
        ["object", 'category']).columns.tolist()

    print(f'Las columnas categoricas son: {len(df_cat)}')

    return df_cat
    
    
def numericas(df):
    df_num=df.select_dtypes(include = \
        ['float64','float64','int64']).columns.tolist()
    print(f'Las columnas numericas son: {len(df_num)}')
    return df_num

def valores_unicos_columna(df,columna):
    unica=df[columna].unique()
    return unica
    

def resumen_inicial(df):
    tabla_resumen=pd.DataFrame(

    {
        'columns':df.columns,
        'tipo de dato': [ df[i].dtype for i in df.columns   ],
        'categorias': [ df[i].nunique()   for i in df.columns ]
    })
    return tabla_resumen

def valores_unicos_categorias(df):
    df_cat = df.select_dtypes(include = \
        ["object", 'category']).columns.tolist()
    for i in df_cat:
        print(f'{i}: {df[i].nunique()}')
    
def nullos(df):
    nullo = df.isnull().sum().reset_index()
    nullo.columns = ['variable', 'conteo']
    nullo=nullo[nullo.conteo!=0].iloc[:,0]
    columna=[ i  for i in nullo ]
    filtro=df[df[columna].isnull().any(axis=1)]
    print(f'la cantidad de nulos es: {sum(df.isnull().sum())}')
    return filtro
    
def duplicados(df):
    print(f'la cantidad de duplicados es: {df.duplicated().sum()}')
    
def transfor_fecha(df, columnas):
    for col in columnas:
        df[col] = pd.to_datetime(df[col])

def caracteristicas_date(df,variable):
    df['YEAR'] = df[variable].dt.year
    df['MONTH'] = df[variable].dt.month
    df['WEEK'] = df[variable].dt.isocalendar().week
    return df



def columnas(df):
    columnas = []
    for col in df.columns:
        columnas.append(col)
        
    return columnas

def vacios(x):
    if x == '  ':
        return np.nan  # Retorna None (nulo)
    else:
        return x
    
import pandas as pd
from sklearn.impute import SimpleImputer

def imputar(df):
    # Almacena los tipos de datos originales
    tipos_originales = df.dtypes
    
    # Crea un imputador que utiliza la estrategia de imputación más frecuente
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')
    
    # Aplica el imputador y convierte el resultado a un DataFrame
    df_imputed = pd.DataFrame(imputer_most_frequent.fit_transform(df), columns=df.columns)
    
    # Restaura los tipos de datos originales para columnas numéricas
    for columna in df.columns:
        if tipos_originales[columna] in ['float64', 'int64']:
            df_imputed[columna] = pd.to_numeric(df_imputed[columna], errors='coerce')
    
    return df_imputed


def tipo_funciones():
    funciones = [func[0] for func in inspect.getmembers(funciones_eda, inspect.isfunction)]
    funciones=[ i for i in funciones]
    return funciones

def tabla__(df,variable):
    tabla=df[variable].value_counts().reset_index()
    tabla['acumulado']=round((tabla['count'].cumsum())/tabla['count'].sum()*100,1)
    #tabla=tabla[tabla['acumulado']<=80]
    return tabla


def shapiro_test(variable):
    data = variable
    stat, p_value = shapiro(data)
    print(f"Estadístico: {stat}")
    print(f"P-valor: {p_value}")
    alpha = 0.05
    if p_value > alpha:
        print("Los datos tienen una distribución normal (no se rechaza H0)")
    else:
        print("Los datos no tienen una distribución normal (se rechaza H0)")

def kolmogorov_smirnov_test(data):
    # Realizar la prueba KS para verificar normalidad
    stat, p_value = kstest(data, 'norm')
    
    # Imprimir los resultados
    print(f"Estadístico KS: {stat}")
    print(f"P-valor: {p_value}")
    
    # Interpretar el resultado
    alpha = 0.05
    
   
    if p_value > alpha:
        print("No se rechaza la hipótesis nula: los datos podrían seguir una distribución normal")
    else:
        print("Se rechaza la hipótesis nula: los datos no siguen una distribución normal")

    
    


def calcular_probabilidad_compra(directa_):
    # Agrupar el historial semanal por las nuevas variables BRAND_NAME y SECTOR_NAME
    historial_semanal = directa_.groupby(
        ['CUSTOMER_ID', 'PRODUCT_ID', 'Semana', 'DIST_CHANNEL_NAME', 'SUB_DIST_CHANNEL_NAME', 'TIPOL_TRANS', 'BRAND_NAME', 'SECTOR_NAME']
    )['QTY'].sum().reset_index()
    
    # Calcular el total de compras por cliente, considerando las nuevas variables
    total_compras_cliente = historial_semanal.groupby(
        ['CUSTOMER_ID', 'Semana', 'DIST_CHANNEL_NAME', 'SUB_DIST_CHANNEL_NAME', 'TIPOL_TRANS', 'BRAND_NAME', 'SECTOR_NAME']
    )['QTY'].sum().reset_index()
    
    # Renombrar la columna QTY a total_compras_cliente
    total_compras_cliente = total_compras_cliente.rename(columns={'QTY': 'total_compras_cliente'})
    
    # Unir el total de compras con el historial semanal
    historial_semanal = historial_semanal.merge(
        total_compras_cliente,
        on=['CUSTOMER_ID', 'Semana', 'DIST_CHANNEL_NAME', 'SUB_DIST_CHANNEL_NAME', 'TIPOL_TRANS', 'BRAND_NAME', 'SECTOR_NAME'],
        how='left'
    )
    
    # Calcular la probabilidad de compra del producto
    historial_semanal['probabilidad_compra_producto'] = (
        historial_semanal['QTY'] / historial_semanal['total_compras_cliente']
    )
    
    # Ordenar el DataFrame por CUSTOMER_ID, PRODUCT_ID y Semana
    historial_semanal = historial_semanal.sort_values(
        by=['CUSTOMER_ID', 'PRODUCT_ID', 'Semana']
    ).reset_index(drop=True)
    
    return historial_semanal






#-------- outliers  --------------# 

def eliminar_outliers_zscore(df, columna, umbral=2):
    """
    Elimina outliers utilizando el método Z-Score en una columna específica de un DataFrame.
    
    :param df: DataFrame sobre el cual se aplicará el filtro
    :param columna: Nombre de la columna en la cual se calculará el Z-Score
    :param umbral: Umbral para definir outliers (por defecto es 2)
    :return: DataFrame filtrado sin outliers
    """
    # Calcular la media y la desviación estándar
    media = df[columna].mean()
    desviacion_std = df[columna].std()

    # Calcular el Z-Score
    z_score = ((df[columna] - media) / desviacion_std).abs()

    # Identificar outliers
    outliers = df[z_score > umbral]
    
    # Filtrar el DataFrame eliminando los outliers
    df_filtrado = df[z_score <= umbral]
    
    return df_filtrado


def eliminar_outliers_iqr(df, columna, multiplicador=1.5):
    

    """
    Elimina outliers utilizando el método del Rango Intercuartílico (IQR) en una columna específica.
    
    :param df: DataFrame sobre el cual se aplicará el filtro
    :param columna: Nombre de la columna en la cual se calculará el IQR
    :param multiplicador: Multiplicador para definir los límites (por defecto es 1.5, pero se puede ajustar)
    :return: DataFrame filtrado sin outliers y DataFrame con outliers
    """
    # Calcular el primer y tercer cuartil
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)

    # Calcular el rango intercuartílico (IQR)
    IQR = Q3 - Q1

    # Definir los límites para identificar outliers
    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR

    # Identificar los outliers
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]

    # Filtrar el DataFrame eliminando los outliers
    df_filtrado = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    
    return df_filtrado

    
    

 
 
 
 
#-------- analisis exploratrio-----------------#   

def graficar_barras_columnas(df, columnas, tipo_agrupacion='count', valor=None, paleta='viridis'):
    """
    Genera gráficos de barras para las columnas especificadas en el DataFrame.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de columnas a graficar.
        tipo_agrupacion (str): Tipo de agrupación ('count' o 'sum').
        valor (str): Nombre de la columna a sumar (solo si tipo_agrupacion es 'sum').
    """
    num_columnas = len(columnas)
    num_filas = math.ceil(num_columnas / 3)
    
    sns.set_style("whitegrid")
 
    
    fig, ax = plt.subplots(num_filas, 3, figsize=(15, 5 * num_filas))
    ax = ax.flatten()
    for i, column in enumerate(columnas):
        datos = df[column].value_counts().reset_index().head(10)
        datos.columns = [column, 'Frecuencia']
        datos=datos.sort_values(by='Frecuencia')
        sns.barplot(data=datos, x=column, y='Frecuencia', ax=ax[i], palette=paleta,hue=column, dodge=False, legend=False)
        ax[i].set_ylabel("Frecuencia")
        ax[i].set_xlabel(column)
        ax[i].set_title(f'Gráfico de Barras de {column}')
        ax[i].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()


    
#-------- analisis exploratrio-----------------#   

def barras_80(df,variable, paleta='viridis'):
   warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")
   datos = df[variable].value_counts().reset_index().head(40)
   datos.columns = [variable, 'Frecuencia']
   datos=datos.sort_values(by='Frecuencia',ascending=False)
   datos['porcentaje']=round((datos['Frecuencia']/datos['Frecuencia'].sum())*100,2)
   datos['Frecuencia_acumulada'] = round(datos['porcentaje'].cumsum(),2)
   
   categoria_80 = datos[datos['Frecuencia_acumulada'] <= 80][variable].iloc[-1]

   plt.figure(figsize=(10, 6))
   sns.barplot(data=datos, x=variable, y='Frecuencia', palette="viridis", dodge=False, legend=False, hue=variable,order=datos[variable])
   plt.ylabel("Frecuencia")
   plt.xlabel(variable)
   plt.title(f'Gráfico de Barras de {variable}')
   plt.xticks(rotation=90)  # Rotar etiquetas si son largas

   plt.axvline(x=list(datos[variable]).index(categoria_80), color='red', linestyle='--', label='80% acumulado')   

   plt.tight_layout()
   plt.show()


def tabla_(df,variable, paleta='viridis',had=5):
    datos = df[variable].value_counts().reset_index().head(had)
    datos.columns = [variable, 'Frecuencia']
    datos=datos.sort_values(by='Frecuencia',ascending=False)
    datos['porcentaje']=round((datos['Frecuencia']/datos['Frecuencia'].sum())*100,2)
    datos['Frecuencia_acumulada'] = round(datos['porcentaje'].cumsum(),2)
    
    plt.figure(figsize=(7, 8))
    sns.barplot(data=datos, x=variable, y='Frecuencia', palette="viridis", dodge=False, legend=False, hue=variable,order=datos[variable])
    plt.ylabel("Frecuencia")
    plt.xlabel(variable)
    plt.title(f'Gráfico de Barras de {variable}')
    plt.xticks(rotation=90)  # Rotar etiquetas si son largas

     

    plt.tight_layout()
    plt.show()
    
    return datos[[variable,'Frecuencia_acumulada']]



#--------- distribucion de variables-----------------#

def plot_normal_distribution(df, variables):
    """
    Genera gráficos de distribución para múltiples características y los compara con una distribución normal teórica.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - variables: Lista de nombres de columnas (str) en df que se quieren analizar.
    """
    # Configuración de subgráficos
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))

    for i, variable in enumerate(variables):
        ax = axes[i] if len(variables) > 1 else axes  # Para manejar el caso de un solo gráfico

        # Histograma con KDE
        sns.histplot(df[variable], kde=True, color="skyblue", bins=30, stat="density", label="Datos", ax=ax)

        # Ajuste a una distribución normal
        media = df[variable].mean()
        desviacion_estandar = df[variable].std()
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, media, desviacion_estandar)

        # Curva de la distribución normal teórica
        ax.plot(x, p, 'r--', linewidth=2, label=f'Distribución  (μ={media:.2f}, σ={desviacion_estandar:.2f})')

        # Título y etiquetas
        ax.set_title(f'{variable} vs. Distribución Normal')
        ax.set_xlabel(variable)
        ax.set_ylabel('Densidad')
        ax.legend()

    plt.tight_layout()
    plt.show()


#-------------box_plot-----------------#
def plot_boxplot_normal_comparison(df, variables):
    """
    Genera gráficos de boxplot para múltiples características y los compara con estadísticas de una distribución normal teórica.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - variables: Lista de nombres de columnas (str) en df que se quieren analizar.
    """
    # Configuración de subgráficos
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))

    for i, variable in enumerate(variables):
        ax = axes[i] if len(variables) > 1 else axes  # Para manejar el caso de un solo gráfico

        # Boxplot
        sns.boxplot(x=df[variable], color="lightgreen", ax=ax)

        # Líneas de la media y desviación estándar de la distribución normal teórica
        media = df[variable].mean()
        desviacion_estandar = df[variable].std()
        ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media (μ={media:.2f})')
        ax.axvline(media - desviacion_estandar, color='blue', linestyle='--', linewidth=1, label=f'σ={desviacion_estandar:.2f}')
        ax.axvline(media + desviacion_estandar, color='blue', linestyle='--', linewidth=1)

        # Título y etiquetas
        ax.set_title(f'{variable} - Boxplot y Media Normal Teórica')
        ax.set_xlabel(variable)
        ax.set_ylabel('Valor')
        ax.legend()

    plt.tight_layout()
    plt.show()



#----------------------graficos de lineas meses--------------------#
def lineas_meses_año(df, variable):
    
    
    df = df.copy()
    
    df['año']=df[variable].dt.year
    df = df[df['año'] > 2019]
    anio=[i for i in df['año'].unique()]
    
    meses = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
    
    plt.figure(figsize=(10, 6))
    
    
    for i in anio:
        data=df[df['año']==i]
        data=data[variable].dt.month.value_counts().reset_index()
        data.sort_values(by=variable,ascending=True)
        
        data['Mes'] = data[variable].map(meses)
        data=data.sort_values(by=variable,ascending=True)
        data=data[['Mes','count']]
        
        sns.lineplot(data=data, x='Mes', y='count', marker='o',label=str(i))
    
    
    sns.lineplot(data=data, x='Mes', y='count', marker='o')
    plt.ylabel("Frecuencia")
    plt.xlabel("Mes")
    plt.title("Frecuencia de ocurrencias por mes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

    
    
#-------------- histograma------------------#

def histograma(df,bi=50):
    plt.hist(df, bins=bi, edgecolor='black')  # 'bins' es el número de barras en el histograma
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title(f'Histograma ')

    # Mostrar el histograma
    plt.show()


#------------- boxplot--------------#

def boxplot(df,variable):
    plt.figure(figsize=(6, 6))

    # Crear el boxplot
    sns.boxplot(y=df[df[variable]!=0][variable])

    # Añadir título y etiqueta
    plt.title("Distribución de dias_diferencia_entrega")
    plt.ylabel("dias_diferencia_entrega")

    # Mostrar el gráfico
    plt.show()

def boxplot_m(df, variables):
    for variable in variables:
        plt.figure(figsize=(6, 6))

        # Crear el boxplot para cada variable
        sns.boxplot(y=df[df[variable] != 0][variable])

        # Añadir título y etiquetas
        plt.title(f"Distribución de {variable}")
        plt.ylabel(variable)

        # Mostrar el gráfico
        plt.show()


#----------- QQ PLOT-------------------#

def qq_plot(variable):
    plt.figure(figsize=(8, 6))
    stats.probplot(variable, dist="norm", plot=plt)
    plt.title("QQ Plot para el Análisis de Normalidad")
    plt.xlabel("Cuantiles Teóricos")
    plt.ylabel("Cuantiles de los Datos")
    plt.grid(True)
    plt.show()
    
    
#-------- scater plot-------#

def scater_plot(df,x,y):
    sns.scatterplot(data=df, x=x, y=y)
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.show()