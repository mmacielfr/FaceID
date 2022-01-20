import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

imagePath = 'imagem_teste.jpg'
#leia a imagem
image = cv2.imread(imagePath)

cascadePath = 'haarcascade_frontalface_default.xml'
#vamos utilizar novamente o classificador provido pelo opencv
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#a detecção de face é feita em imagem em escala de cinza
#vamos convert a imagem original
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#vamos utilizar o método detectMultiScale com 
faces = faceCascade.detectMultiScale(
    gray, 
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
#imprimindo quantas faces foram encontradas
print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow("", image)
  cv2.waitKey(0)