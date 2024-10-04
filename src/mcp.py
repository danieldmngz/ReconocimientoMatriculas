import cv2
import numpy as np
import easyocr  # Importar EasyOCR
from ultralytics import YOLO
import matplotlib.pyplot as plt
import re  # Importar módulo de expresiones regulares

# Cargar modelo YOLO
model = YOLO(r'C:\Users\Daniel.Dominguez\Documents\Data\Data\runs\detect\train2\weights\best.pt')

# Cargar imagen
img_path = r'C:\Users\Daniel.Dominguez\source\repos\Python\ReconocimientoMatriculas\ReconocimientoMatriculas\Imagenes\Matriculas\230602\E1,0000000028,230602,060114,FQW762.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Error al leer la imagen.")
    exit()

# Realizar la detección con YOLO
results = model.predict(img)

placa = None  # Inicializar la variable placa

# Mostrar resultados de YOLO
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar el rectángulo en verde
        
        # Extraer la región de la placa
        placa = img[y1:y2, x1:x2]  # Cambiar a `img` ya que estamos usando YOLO

# Verificar si se detectó una placa
if placa is None:
    print("No se detectaron placas.")
    exit()

# Convertir la placa a escala de grises
placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

# Procesar la placa
canny = cv2.Canny(placa_gray, 50, 150)
binaria = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Hough Transform para detectar líneas
edges = cv2.Canny(placa_gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=1, maxLineGap=100)

lineas_horizontales = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        angulo = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if np.abs(angulo) < 20 or np.abs(angulo - 180) < 20:
            lineas_horizontales.append((x1, y1, x2, y2))

if lineas_horizontales:
    angulo_promedio = np.mean([np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lineas_horizontales]) * 180 / np.pi
    centro = (placa.shape[1] // 2, placa.shape[0] // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_promedio, 1.0)
    imagen_enderezada = cv2.warpAffine(placa, matriz_rotacion, (placa.shape[1], placa.shape[0]))
else:
    print("No se encontraron líneas horizontales.")
    imagen_enderezada = placa  # No hay rotación

# Procesar para mejorar la detección de texto
if len(imagen_enderezada.shape) == 2:  # Si es una imagen en escala de grises
    imagen_enderezada = cv2.cvtColor(imagen_enderezada, cv2.COLOR_GRAY2BGR)

gray2 = cv2.cvtColor(imagen_enderezada, cv2.COLOR_BGR2GRAY)
framee = cv2.bilateralFilter(gray2, 15, 20, 17)
bord2 = cv2.Canny(framee, 50, 150, apertureSize=7)

# Transformación de perspectiva
dimensiones = framee.shape
h, w = dimensiones[0], dimensiones[1]

pts1 = np.float32([[w * 0.01, h * 0.18], [w * 0.80, h * 0.10], [w * 0.10, h * 0.80], [w * 0.89, h * 0.89]])
pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(framee, M, (w, h))

# Reconocimiento de texto con EasyOCR
reader = easyocr.Reader(['es'])  # Usar español como idioma
resultados = reader.readtext(imagen_enderezada)

# Procesar resultados de EasyOCR
patron_placa = re.compile(r'^[A-Z]{3}-\d{3}$')  # Expresión regular para validar placas en formato ABC-123

placa_detectada = None  # Inicializar la variable para almacenar la placa detectada

for (bbox, texto, probabilidad) in resultados:
    # Limpiar texto de caracteres no alfanuméricos, permitiendo guiones
    texto_limpio = ''.join(filter(lambda x: x.isalnum() or x == '-', texto)).strip().upper()  # Convertir a mayúsculas
    print(f'Texto detectado: {texto_limpio}')  # Imprimir texto detectado para depuración
    
    # Validar la placa
    if patron_placa.match(texto_limpio):  # Verificar si el texto coincide con el patrón
        placa_detectada = texto_limpio  # Almacenar la placa detectada
        break  # Salir del bucle si se encuentra una placa válida


# # Imprimir el texto detectado
# if placa_detectada:
#     print('Matricula: ', placa_detectada)
# else:
#     print("No se detectó una placa válida.")

# Escalar la imagen enderezada a 4K para una mejor visualización
imagen_enderezada_4k = cv2.resize(imagen_enderezada, (800, 600), interpolation=cv2.INTER_LINEAR)

# Mostrar la imagen final con detecciones de YOLO usando Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB
plt.axis('on')  # Ocultar los ejes
plt.title('Detección de placa con YOLO Redes neuronales Convolucionales')
plt.show()

# Mostrar la imagen enderezada en 4K
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(imagen_enderezada_4k, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB
plt.axis('on')  # Ocultar los ejes
plt.title('Imagen Enderezada de la Placa en 4K')
plt.show()
