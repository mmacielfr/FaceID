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
  #CRIANDO A IMAGEM PARA AUMENTAR O BRILHO APENAS DO VERMELHO
  red = np.zeros_like(ROI); #CRIAR IMAGEM DO TAMANHO DA ROI
  red[:,:] = [0,0,100]; # (b,g,r)
   
  #AUMENTE O BRILHO DO CANAL VERMELHO DA ROI 
  avermelhado = cv2.add(ROI, red);
  #LIMITE O VALOR PARA 255
  avermelhado[avermelhado > 255] = 255
   
  #'colando' a face avermelhada na foto original
  #ps.: não é a única forma. 
  #Pode ser feito o processamento direto na imagem original
  imageAvermelhado = image.copy()
  imageAvermelhado[y:y+h, x:x+w] = avermelhado
  cv2.imshow("", imageAvermelhado)
  cv2.waitKey()