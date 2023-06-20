import cv2
import pytesseract





def save_textInFile(image_path, text):
    with open(image_path.split('.')[0] + '.txt', 'w') as f:
        f.write(text)


def detect_text(image_path, gray):
    # Загрузка изображения
    image = cv2.imread(image_path)
    #Распознавание границ
    edges = cv2.Canny(gray, 30, 150)

    #Находим контуры на изображении 
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Для каждого найденного контура рисуем ограничивающий прямоугольник вокруг него
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return image

def enhance_text(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    #Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Применяет пороговое значение к изображению в оттенках серого (для отделения текста), используя функцию с двумя флагами
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    #Создаем ядро прямоугольной формы размером 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    #Применяем расширение к изображению с пороговым значением
    dilate = cv2.dilate(thresh, kernel, iterations=0)
    
    return dilate

def recognize_text(image_path, traindedModelsPath, language):


    enhance_result = enhance_text(image_path)

    cv2.imshow('Detected Text', detect_text(image_path, enhance_result))
    cv2.imshow('Enhanced Text', enhance_text(image_path))

    # Применение OCR для распознавания текста
    config = f"--oem 1 --psm 3 -l {language} --tessdata-dir {traindedModelsPath}"
    text = pytesseract.image_to_string(enhance_result, config=config)

    # Вывод распознанного текста
    print(text)

    #Сохранение результата в файл
    save_textInFile(image_path, text)
    cv2.waitKey(0)


image_path = "1.png"
traindedModelsPath = 'tessdata'

recognize_text(image_path, traindedModelsPath, "rus+eng")