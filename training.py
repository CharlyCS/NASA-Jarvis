# Importa las bibliotecas necesarias
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Carga el archivo JSON 
intents = json.loads(open('chatbot-beta/intents1.json',encoding='utf-8').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializa listas y variables
words = []        # Almacenará todas las palabras en los patrones
classes = []      # Almacenará las etiquetas de intención
documents = []    # Almacenará los patrones de palabras junto con sus etiquetas
ignore_letters = ['?','!','¿','.',',']  # Lista de caracteres a ignorar

# Recorre las intenciones y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza el patrón en palabras individuales
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Agrega las palabras a la lista 'words'
        documents.append((word_list, intent['tag']))  # Agrega el patrón y la etiqueta a 'documents'
        if intent['tag'] not in classes:
            classes.append(intent["tag"])  # Agrega la etiqueta a 'classes' si no está presente

# Lematiza y filtra las palabras, luego las ordena y las almacena en archivos binarios
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Elimina duplicados y ordena las palabras
# Guarda las pattents en un archivo binario
pickle.dump(words, open('chatbot-beta/words.pkl', 'wb'))  
# Guarda las etiquetas en un archivo binario
pickle.dump(classes, open('chatbot-beta/classes.pkl','wb'))  

# Inicializa las listas para entrenamiento y crea las representaciones de bolsas y salidas esperadas
training = []  # Almacenará las representaciones
output_empty = [0]*len(classes)  # Crea una lista de ceros 
for document in documents:
    bag = []  # Representación de bolsa para el documento actual
    word_patterns = document[0]  # Obtiene los patrones 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] 
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)  
    output_row = list(output_empty)  # Crea una copia de la lista de salidas esperadas
    output_row[classes.index(document[1])] = 1  
    training.append([bag, output_row])  

# Mezcla aleatoriamente los datos de entrenamiento y los convierte en un arreglo NumPy
random.shuffle(training)
training = np.array(training)

# Imprime el arreglo de datos de entrenamiento
print(training)

#Reparte los datos para pasarlos a la red
train_x = list(training[:,0])
train_y = list(training[:,1])

#Creamos la red neuronal
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Creamos el optimizador y lo compilamos
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save("chatbot-beta/chatbot_model.h5", train_process)




