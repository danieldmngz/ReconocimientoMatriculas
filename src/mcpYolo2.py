import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar el modelo YOLOv8 localmente
model = YOLO(r'C:\Users\Daniel.Dominguez\Documents\Data\Data\runs\detect\train2\weights\best.pt')

# Cargar la imagen
img_path = r'C:\Users\Daniel.Dominguez\source\repos\Python\ReconocimientoMatriculas\ReconocimientoMatriculas\Imagenes\Matriculas\230602\E1,0000000000,230602,100249,FUM670.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Error al leer la imagen.")
    exit()

# Realizar la detección
results = model.predict(img)

# Extraer las coordenadas de las detecciones
boxes = results[0].boxes.xyxy.cpu().numpy()  # Formato xyxy (x1, y1, x2, y2)
confs = results[0].boxes.conf.cpu().numpy()  # Confianza de la detección

# Verificar si se detectaron placas
if len(boxes) == 0:
    print("No se detectaron placas.")
else:
    for i in range(len(boxes)):
        box = boxes[i]  # Obtener las coordenadas de la caja
        conf = confs[i]  # Obtener la confianza

        print(f"Valores en box: {box}")  # Imprime los valores en box para verificar
        if conf > 0.5:  # Umbral de confianza
            x1, y1, x2, y2 = map(int, box)
            print(f'Dibujando rectángulo en: ({x1}, {y1}), ({x2}, {y2})')  # Verifica las coordenadas
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar el rectángulo

            # Extraer la imagen de la placa
            plate = img[y1:y2, x1:x2]

            # Mostrar la imagen de la placa antes del reconocimiento
            cv2.imshow('Placa Original', plate)
            cv2.waitKey(500)  # Esperar 500 ms para ver la imagen

            # Preprocesamiento de la imagen de la placa
            gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_plate = clahe.apply(gray_plate)
            edges = cv2.Canny(gray_plate, 30, 200)
            kernel = np.ones((3, 3), np.uint8)
            dilated_plate = cv2.dilate(edges, kernel, iterations=1)
            binaria_plate = cv2.threshold(dilated_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Invertir la imagen
            binaria_plate = cv2.bitwise_not(binaria_plate)

            # Mostrar la imagen preprocesada de la placa
            cv2.imshow('Placa procesada', binaria_plate)
            cv2.waitKey(500)  # Esperar 500 ms para ver la imagen

            # Reconocimiento de texto con Tesseract
            text = pytesseract.image_to_string(binaria_plate, config='--oem 3 --psm 8 -l spa')
            text = text.strip()

            print(f'Texto detectado: "{text}"')  # Imprimir el texto detectado

            if text:
                print(f'Placa detectada: {text}')
                cv2.destroyAllWindows()
                break  # Terminar después de detectar la primera placa correctamente

# Mostrar la imagen con las detecciones
cv2.imshow('Deteccion de Matricula', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

