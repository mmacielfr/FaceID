"""
Original file is located at
    https://colab.research.google.com/drive/1yifJA0F4v92UB6RrzU8KQDftDhx9DBki
"""

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

# Desenhando o retangulo ao redor de cada face encontrada
#(0,255,0) é a cor do retangulo (verde)
#2 é a grossura da linha
#w é a largura do quadrado da face
#h é a altura
#x é o ponto inicial no eixo horizontal
#y é o ponto inicial no eixo vertical

for (x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow(image)
  cv2.waitKey(0)
"""
for (x, y, w, h) in faces:  
  #RECORTANDO APENAS A REGIÃO DA FACE
  #PERCEBA QUE NO RECORTE, OS EIXOS SÃO INVERSOS
  ROI = image[y:y+h, x:x+w,:]
  #MOSTRANDO A REGIAO DA FACE
  cv2.imshow(ROI)
  cv2.waitKey(0)   
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
  cv2.imshow(imageAvermelhado)
  cv2.waitKey()

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
  cv2.imshow(imageEmbacado)
  cv2.waitKey()
"""