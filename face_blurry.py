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
  #RECORTANDO APENAS A REGIÃO DA FACE
  #PERCEBA QUE NO RECORTE, OS EIXOS SÃO INVERSOS
  ROI = image[y:y+h, x:x+w,:] 

  #crie um filtro (no caso esse embaça a image) e aplique
  #kernel = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1 ]])
  kernel = np.ones((5,5),np.float32)/25
  embacado = cv2.filter2D(ROI,-1,kernel)
  #LIMITE O VALOR PARA 255
  embacado[embacado > 255] = 255
   
  #'colando' a face avermelhada na foto original
  #ps.: não é a única forma. 
  #Pode ser feito o processamento direto na imagem original
  imageEmbacado = image.copy()
  imageEmbacado[y:y+h, x:x+w] = embacado
  cv2.imshow("", imageEmbacado)
  cv2.waitKey()