import cv2
import os
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier('libs/haarcascade_frontalface_default.xml')

# Crear el modelo de reconocimiento facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar rostros permitidos
allowed_faces = []
allowed_names = []

# Ruta de la carpeta de rostros permitidos
allowed_folder = 'permitidos/'

# Cargar imágenes de la carpeta "permitidos"
for filename in os.listdir(allowed_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(allowed_folder, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

        for (x, y, w, h) in faces:
            # Guardar la imagen del rostro y su nombre
            allowed_faces.append(gray_image[y:y+h, x:x+w])  # Guardar solo el rostro
            allowed_names.append(os.path.splitext(filename)[0])  # Guardar el nombre sin extensión

# Entrenar el modelo con los rostros permitidos
labels = np.arange(len(allowed_names))  # Crear etiquetas para cada rostro
recognizer.train(allowed_faces, labels)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extraer el rostro detectado
        detected_face = gray[y:y + h, x:x + w]

        # Reconocer el rostro usando el modelo LBPH
        label, confidence = recognizer.predict(detected_face)

        # Verificar la confianza
        if confidence < 100:  # Umbral de confianza ajustado
            name = allowed_names[label]
        else:
            name = "Desconocido"

        # Mostrar el nombre en la imagen
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Reconocimiento Facial', img)
    k = cv2.waitKey(30)
    if k == 27:  # 27 es el ascionamiento para esc (salir)
        break

cap.release()
cv2.destroyAllWindows()