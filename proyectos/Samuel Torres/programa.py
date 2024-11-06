#---------------------------------------------------------------------------------------------------------------
#---------------------------------------Carga de librerias -----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer # cambiar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import skfuzzy as fuzz
import time
import re

start_execution = time.time()

#nltk.download("all")


print("lectura de datos")
df = pd.read_csv('test_data.csv')
print("lectura exitosa")
original = df['sentence']
#********************************************************************************************
#*********************** Preprocesamiento de los datos **********************************************************
#********************************************************************************************

#prepocesamiento de los tweets
def preprocess_text(text):
    """
    Esta funcion preprocesa los datos (tweets)
    
    recibe como parametro los tweets y elimina las URLS,@,_,www, letras sueltas
    """
    tokens = word_tokenize(text.lower())
    
    tokens_ = [ re.sub(r'http[s]?://\S+', '', token) for token in tokens]
    tokens = [ re.sub(r' www\S+', '', token) for token in tokens_]
    tokens_ = [ re.sub(r'@\S+', '', token) for token in tokens]
    tokens = [ re.sub(r'[^\w\s]|[\d]', ' ', token) for token in tokens_]
    tokens_ = [ re.sub(r'\s\s+', ' ', token) for token in tokens]
    tokens = [ re.sub(r'_\S+', '', token) for token in tokens_]
    tokens_cleaned = [re.sub(r'^[a-zA-Z]$', '', token) for token in tokens]
    filtered_tokens = [token for token in tokens_cleaned if token not in stopwords.words('english')]

  #lematizar los tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return ' '.join(lemmatized_tokens)

#limpieza de los tweets
print("Comienzo del preproceso de datos")
df['sentence'] = df['sentence'].apply(preprocess_text)
positivos = df[df['sentiment']==1]
print("finalizacion del preprocesamiento")
#carga de los tweets limpiados
tweets_df = pd.DataFrame(df['sentence'])



#******************************************************************************************************************
#***************************************CALCULO DE LOS PUNTAJES****************************************************
#******************************************************************************************************************

sia = SentimentIntensityAnalyzer()

#funcion para obtener el puntaje
def get_sentiment_score(text):
    scores = sia.polarity_scores(text)
    scores['pos'] = round(scores['pos'], 2)
    scores['neg'] = round(scores['neg'], 2)
    return scores['pos'],scores['neg']

tweets_df[['puntaje_positivo','puntaje_negativo']]= tweets_df['sentence'].apply(get_sentiment_score).apply(pd.Series)
tweets_df.to_csv('datos.csv', index=False)

#******************************************************************************************************************
#****************************CALCULO DE Las funciones de membresia*************************************************
#******************************************************************************************************************

#Halla los min, max y medio de los puntajes
min_pos =  tweets_df['puntaje_positivo'].min()
max_pos =  tweets_df['puntaje_positivo'].max()
min_neg =  tweets_df['puntaje_negativo'].min()
max_neg =  tweets_df['puntaje_negativo'].max()
mid_pos = (min_pos+max_pos)/2
mid_neg = (min_neg+max_neg)/2

x_p = np.arange(0, 1, 0.001)
x_n = np.arange(0, 1, 0.001)
x_op= np.arange(0,10,0.001)

low_pos = fuzz.trimf(x_p, [min_pos, min_pos, mid_pos])
medium_pos = fuzz.trimf(x_p, [min_pos, mid_pos, max_pos])
high_pos = fuzz.trimf(x_p, [mid_pos, max_pos, max_pos])
low_neg = fuzz.trimf(x_n, [min_neg, min_neg, mid_neg])
medium_neg = fuzz.trimf(x_n, [min_neg, mid_neg, max_neg])
high_neg = fuzz.trimf(x_n, [mid_neg, max_neg, max_neg])
# calculo de las funciones de membresia de los activadores
op_neg = fuzz.trimf(x_op, [0, 0, 5])
op_neu = fuzz.trimf(x_op, [0, 5, 10])
op_pos = fuzz.trimf(x_op, [5, 10, 10])



#variable para almacenar los datos para guardar en el csv
tweets = []
tweets_positivos = []
tweets_negativos = []
tweets_sentimiento_calculado = []
tweets_sentimiento = []
#Variables para medir el tiempo de fuzzificacion y defuzzificacion
time_fuzz = []
time_defuzz = []
time_total =[]

positivos =0
negativos =0
neutrales = 0


for row in tweets_df.itertuples(index=False):
    
    
    start_time_fuzz= time.time()
    
    #almacenamiento de las valores
    tweet = row.sentence
    puntaje_positivo = row.puntaje_positivo
    puntaje_negativo = row.puntaje_negativo
    
    #calculo de los valores de pertenencia por la funcion de membresia de los niveles positivos
    nivel_low_pos = fuzz.interp_membership(x_p,low_pos,puntaje_positivo)
    nivel_mid_pos = fuzz.interp_membership(x_p,medium_pos,puntaje_positivo)
    nivel_high_pos = fuzz.interp_membership(x_p,high_pos,puntaje_positivo)
    #calculo de los valores de pertenencia por la funcion de membresia de los niveles negativos
    nivel_low_neg = fuzz.interp_membership(x_n,low_neg,puntaje_negativo)
    nivel_mid_neg = fuzz.interp_membership(x_n,medium_neg,puntaje_negativo)
    nivel_high_neg = fuzz.interp_membership(x_n,high_neg,puntaje_negativo)
    #Calculo de las reglas de mandani
    WR1 = np.fmin(nivel_low_pos,nivel_low_neg )
    WR2 = np.fmin(nivel_mid_pos,nivel_low_neg)
    WR3 = np.fmin(nivel_high_pos,nivel_low_neg)
    WR4 = np.fmin(nivel_mid_neg,nivel_low_pos)
    WR5 = np.fmin(nivel_mid_neg,nivel_mid_pos)
    WR6 = np.fmin(nivel_mid_neg,nivel_high_pos)
    WR7 = np.fmin(nivel_high_neg,nivel_low_pos)
    WR8 = np.fmin(nivel_high_neg,nivel_mid_pos)
    WR9 = np.fmin(nivel_high_neg,nivel_high_pos)
    
    #calculo de las reglas de agregacion
    Wneg = np.fmax(WR4,WR7)
    Wneg = np.fmax(Wneg,WR8)
    Wneu = np.fmax(WR1,WR5)
    Wneu = np.fmax(Wneu,WR9)

    Wpos = np.fmax(WR2,WR3)
    Wpos = np.fmax(Wpos,WR6)
    
    op_activation_low = np.fmin(Wneg,op_neg)
    op_activation_med = np.fmin(Wneu,op_neu)
    op_activation_high =np.fmin(Wpos,op_pos)
    agregado = np.fmax(op_activation_low,np.fmax(op_activation_med, op_activation_high))
    execution_time_fuzz =time.time()-start_time_fuzz
    
    
    #******************************************************************************************
    #***********************************DESFUZZIFICACION **********************************************
    #******************************************************************************************
    
    time_fuzz += [execution_time_fuzz]
    
    start_time_defuzz = time.time()
    
    output = round( fuzz.centroid( x_op, agregado) , 2 )
    respuesta = ''
    
    if 0<= output <= 3.33 :
        respuesta = 'NegativO'
        negativos += 1
    elif 3.33 < output <= 6.66:
        respuesta = 'Neutro'
        neutrales += 1
    else:
        respuesta = 'Positivo'
        positivos += 1
    execution_time_defuzz = time.time()-start_time_defuzz
    
    #almacena los valores en el array de los datos
    time_defuzz += [execution_time_defuzz]
    time_total += [execution_time_fuzz+execution_time_defuzz]
    tweets += [tweet]
    tweets_positivos += [puntaje_positivo]
    tweets_negativos += [puntaje_negativo]
    tweets_sentimiento += [respuesta]
    tweets_sentimiento_calculado += [output]
    


execution_time_total = time.time() - start_execution

#creacion de un diccionario que almacene los datos
datos = {
    'tweet_original':original,
    'tweet_preprocesado': tweets,
    'puntaje_positivo':tweets_positivos,
    'puntaje_negativo':tweets_negativos,
    'sentimiento_calculado': tweets_sentimiento_calculado,
    'sentimiento': tweets_sentimiento,
    'tiempo_fuzz': time_fuzz,
    'tiempo_defuzz':time_defuzz,
    'tiempo_total': time_total
}
#creacion del nuevo data frame
tweet_df= pd.DataFrame(datos)
tweet_df.to_csv('datos.csv',sep=';', index=False)

print(f'numero de tweets positivos: {positivos} porcentaje {round(positivos/len(tweets_df['sentence'])*100)}%')
print(f'numero de tweets neutrales: {neutrales} porcentaje {round(neutrales/len(tweets_df['sentence'])*100)}%')
print(f'numero de tweets negativos: {negativos} porcentaje {round(negativos/len(tweets_df['sentence'])*100)}%')
print(f'tiempo de ejecucion total : {round(execution_time_total,4)}')
print(f'tiempo de ejecucion promedio : {round(execution_time_total/len(tweets_df['sentence']),4)}')

