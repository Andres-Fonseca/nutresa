from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import warnings

import joblib
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from category_encoders import TargetEncoder
import joblib

def modelos_func_clasificacion(df, modelos, x_, y_):
    warnings.filterwarnings("ignore", category=FutureWarning)
    probabilidades_final_df = pd.DataFrame()
    dr = pd.DataFrame()
    modelos_entrenados = []

    # Definir X e y
    X = df.drop(x_, axis=1)
    y = df[y_]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)
    db_path = '/content/drive/MyDrive/ce/EXCEL/'

    # Aplicar target encoding en CUSTOMER_ID y PRODUCT_ID
    encoder = TargetEncoder(cols=['CUSTOMER_ID', 'PRODUCT_ID'])
    X_train[['CUSTOMER_ID', 'PRODUCT_ID']] = encoder.fit_transform(X_train[['CUSTOMER_ID', 'PRODUCT_ID']], y_train)
    X_test[['CUSTOMER_ID', 'PRODUCT_ID']] = encoder.transform(X_test[['CUSTOMER_ID', 'PRODUCT_ID']])

    # Guardar el encoder para uso futuro
    joblib.dump(encoder, f'{db_path}target_encoder.pkl')

    # Aplicar pd.get_dummies al conjunto de entrenamiento y guardar las columnas
    X_train = pd.get_dummies(X_train)
    columnas_entrenamiento = X_train.columns

    # Guardar la lista de columnas para referencia futura
    with open(f'{db_path}columnas_entrenamiento.pkl', 'wb') as f:
        joblib.dump(columnas_entrenamiento, f)

    # Aplicar pd.get_dummies al conjunto de prueba y reindexar para que coincidan las columnas
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=columnas_entrenamiento, fill_value=0)

    # Escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar el escalador en Google Drive
    joblib.dump(scaler, f'{db_path}scaler.pkl')

    # Loop a través de los modelos
    for i in modelos:
        if i == 'SVC()':
            nombre = 'SVC'
            modelo = SVC(probability=True, random_state=49150)
        elif i == 'RandomForestClassifier()':
            nombre = 'RandomForestClassifier'
            modelo = RandomForestClassifier(class_weight='balanced', random_state=49150)
        elif i == 'DecisionTreeClassifier()':
            nombre = 'DecisionTreeClassifier'
            modelo = DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=49150)
        elif i == 'XGBClassifier()':
            nombre = 'XGBClassifier'
            modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=49150)
        elif i == 'LGBMClassifier()':
            nombre = 'LGBMClassifier'
            modelo = LGBMClassifier(random_state=49150, verbose=-1)
        elif i == 'AdaBoostClassifier()':
            nombre = 'AdaBoostClassifier'
            modelo = AdaBoostClassifier(random_state=49150)
        elif i == 'GradientBoostingClassifier()':
            nombre = 'GradientBoostingClassifier'
            modelo = GradientBoostingClassifier(random_state=49150)
        elif i == 'LogisticRegression()':
            nombre = 'LogisticRegression'
            modelo = LogisticRegression(class_weight='balanced', random_state=49150)
        elif i == 'HistGradientBoostingClassifier()':
            nombre = 'HistGradientBoostingClassifier'
            modelo = HistGradientBoostingClassifier(random_state=49150)
        elif i == 'GaussianNB()':
            nombre = 'GaussianNB'
            modelo = GaussianNB()
        else:
            raise ValueError(f"Modelo {i} no está soportado.")

        # Ajustar el modelo
        modelo.fit(X_train_scaled, y_train)

        # Guardar el modelo entrenado en Google Drive
        modelo_path = f'{db_path}{nombre}_entrenado.pkl'
        joblib.dump(modelo, modelo_path)
        modelos_entrenados.append((nombre, modelo_path))

        # Obtener probabilidades si el modelo tiene el método predict_proba
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para la clase positiva
            
            # Crear DataFrame de probabilidades con CUSTOMER_ID y PRODUCT_ID
            probabilidades_df = pd.DataFrame({
                'CUSTOMER_ID': X_test['CUSTOMER_ID'].values,
                'PRODUCT_ID': X_test['PRODUCT_ID'].values,
                'Semana': X_test['Semana'].values,
                'probabilidad_compra': y_prob
            })
            
            # Agregar nombre del modelo para identificación
            probabilidades_df['model'] = nombre
            
            # Concatenar probabilidades en el DataFrame final
            probabilidades_final_df = pd.concat([probabilidades_final_df, probabilidades_df], ignore_index=True)
        
        # Calcular métricas
        y_pred = modelo.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy_train = accuracy_score(y_train, modelo.predict(X_train_scaled))

        # Crear un DataFrame con los resultados de métricas
        nuevoModelo = pd.DataFrame({
            'model': [nombre],
            'accuracy_test': [accuracy],
            'precision_test': [precision],
            'recall_test': [recall],
            'f1_score_test': [f1],
            'accuracy_train': [accuracy_train]
        })

        # Usar pd.concat para añadir el nuevo modelo al DataFrame de métricas
        dr = pd.concat([dr, nuevoModelo], ignore_index=True)

    # Guardar DataFrame de métricas y probabilidades en archivos CSV en Google Drive
    dr.to_csv(f'{db_path}resultados_modelos_clasificacion.csv', index=False)
    probabilidades_final_df.to_csv(f'{db_path}probabilidades_modelos.csv', index=False)

    return dr, probabilidades_final_df





#---------------feature importanfce ----------------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, TheilSenRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

def importance_1_clasificacion(modelos, df, y_column='dias_diferencia_entrega'):
    # Convertir variables categóricas a variables dummies
    df = pd.get_dummies(df, drop_first=True)

    # Definir X e y
    X = df.drop(y_column, axis=1)
    y = df[y_column]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49150, test_size=0.2)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    resultados_globales = []

    # Loop a través de los modelos
    for i in modelos:
        # Seleccionar y asignar el modelo según el nombre
        if i == 'SVC()':
            nombre = 'SVC'
            modelo = SVC()
        elif i == 'RandomForestClassifier()':
            nombre = 'RandomForestClassifier'
            modelo = RandomForestClassifier(random_state=49150)
        elif i == 'DecisionTreeClassifier()':
            nombre = 'DecisionTreeClassifier'
            modelo = DecisionTreeClassifier(random_state=49150)
        elif i == 'XGBClassifier()':
            nombre = 'XGBClassifier'
            modelo = XGBClassifier(random_state=49150)
        elif i == 'LGBMClassifier()':
            nombre = 'LGBMClassifier'
            modelo = LGBMClassifier(random_state=49150, verbose=-1)
        elif i == 'AdaBoostClassifier()':
            nombre = 'AdaBoostClassifier'
            modelo = AdaBoostClassifier(random_state=49150)
        elif i == 'GradientBoostingClassifier()':
            nombre = 'GradientBoostingClassifier'
            modelo = GradientBoostingClassifier(random_state=49150)
        elif i == 'LogisticRegression()':
            nombre = 'LogisticRegression'
            modelo = LogisticRegression(random_state=49150)
        elif i == 'HistGradientBoostingClassifier()':
            nombre = 'HistGradientBoostingClassifier'
            modelo = HistGradientBoostingClassifier(random_state=49150)
        else:
            print(f"Modelo {i} no está soportado para importancia de características.")
            continue

        # Entrenar el modelo
        modelo.fit(X_train_scaled, y_train)

        # Verificar si el modelo tiene el atributo feature_importances_
        if hasattr(modelo, 'feature_importances_'):
            importances = modelo.feature_importances_
            feature_importances_df = pd.DataFrame({
                'Característica': X.columns,
                'Importancia': importances
            })

            # Ordenar el DataFrame por importancia de mayor a menor y calcular acumulado
            feature_importances_df = feature_importances_df.sort_values(by='Importancia', ascending=False)
            feature_importances_df['acumulado'] = round(feature_importances_df['Importancia'].cumsum(), 2)
            feature_importances_df['Importancia'] = round(feature_importances_df['Importancia'], 2)

            # Extraer la variable global para análisis agregado
            feature_importances_df['Variable_Global'] = feature_importances_df['Característica'].str.split('_').str[0]
            global_importance = feature_importances_df.groupby('Variable_Global')['Importancia'].sum().reset_index()
            global_importance = global_importance.sort_values(by='Importancia', ascending=False)
            
            # Agregar nombre del modelo para identificar
            global_importance['Modelo'] = nombre
            resultados_globales.append(global_importance)
            
            # Calcular y mostrar Accuracy en el conjunto de prueba
            y_pred = modelo.predict(X_test_scaled)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            print(f"Accuracy del modelo {nombre}: {accuracy}")

    # Concatenar todos los resultados en un solo DataFrame
    if resultados_globales:
        importancia_total_df = pd.concat(resultados_globales, ignore_index=True)
        return importancia_total_df
    else:
        print("No se encontraron modelos con importancia de características.")
        return None



#------ resultados----- #

def resultados_(datos):
    resultados=pd.DataFrame()
    conteo=0
    for i in datos:
        conteo+=1
        result=i.iloc[:1]
        result['modelo']=f'modelo_{conteo}'
        resultados = pd.concat([resultados, result], ignore_index=True)
        resultados=resultados.sort_values(by='r2_test', ascending=False)
    return print(resultados)


