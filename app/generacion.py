import pickle
import keras
import numpy as np
# Claves diccionario metadatos
PALABRAS=10
PALABRA_IDX=11
IDX_PALABRA=12
ANT_PAL=13
SECUENCIAS=14
SIG_PAL=15
ARTISTA = 16

LONG_SEC = 10

# Constantes
URL_MODELOS = "./app/modelos"
ARCHIVO_POSTENTRENO = "datos_postentreno.pkl"
SUF_MODELOS = ".h5"
ARCHIVO_MODELOS = "modelo" + SUF_MODELOS

def get_semilla_random(datos): # función que carga una semilla aleatoria
    seed_idx = np.where(np.array(datos[ANT_PAL])=="\n")[0] # Indices de secuencias que son comienzos de verso, para generar más realistamente.
    idx_semilla = np.random.choice(seed_idx, size = 1)[0] # Elección de índice de semilla aleatoria
    semilla = datos[SECUENCIAS][idx_semilla] # Se consigue la semilla en forma de string
    return semilla
def get_datos_modelo(url_modelo): # Conseguir los metadatos de entrenamiento del modelo
    return pickle.load(open(url_modelo + "/" + ARCHIVO_POSTENTRENO, "rb"))

def sampleo(predicciones, temp=0.4): # Función utilitaria que samplea un índice de un array de probabilidades aplicando el sampleo por temperatura
    predicciones = np.asarray(predicciones).astype('float64') # Cambiamos de tipo
    predicciones = np.log(predicciones) / temp # Dividimos por temperatura, cuanto menor temperatura, mayor confianza tendrá el modelo en las probabilidades altas
    exp_preds = np.exp(predicciones)
    predicciones = exp_preds / np.sum(exp_preds) # Hacemos softmax
    probas = np.random.multinomial(1, predicciones, 1) # Sampleo de distribución multinominal definida por las probabilidades en "predicciones"
    return np.argmax(probas)

def generar_liricas(url_modelo, datos, semilla, num_palabras=50, diversidad=0.4):
    modelo = keras.models.load_model(url_modelo+ "/" + ARCHIVO_MODELOS)
    secuencia = semilla
    info = ""
    texto_generado = ""
    info += f"Generando texto con letras de [{datos[ARTISTA]}]\n"
    info += f"----- Diversidad: {diversidad}\n"
    info += f"----- Nº Palabras generadas: {diversidad}\n"
    info += f"----- A partir de la semilla: {' '.join(secuencia)}\n"
    texto_generado += ' '.join(secuencia)

    for i in range(num_palabras): #Se generaran 50 tokens
        # Transformamos la secuencia de palabras en una matriz de entrada para modelo
        x_pred = np.zeros((1, LONG_SEC))
        for t, palabra in enumerate(secuencia):
            x_pred[0, t] = datos[PALABRA_IDX][palabra]

        # Hacemos la predicción
        predicciones = modelo.predict(x_pred, verbose=0)[0]
        
        # Hacemos un sampleo sobre el vector de probabilidades que devuelve el modelo
        siguiente_idx = sampleo(predicciones, diversidad)
        siguiente_palabra = datos[IDX_PALABRA][siguiente_idx]

        secuencia = secuencia[1:]
        secuencia.append(siguiente_palabra)

        texto_generado += f" {siguiente_palabra}"
    return info, texto_generado

def devuelve_generacion_aleatoria(): # Función básica que devuelve letra generada de manera completamente aleatoria. Tambien devuelve info sobre los parámetros de la generación. PARA PROBAR
	url_modelo = URL_MODELOS + "/oscart"
	datos = get_datos_modelo(url_modelo) 
	semilla = get_semilla_random(datos)
	return generar_liricas(url_modelo, datos, semilla)
