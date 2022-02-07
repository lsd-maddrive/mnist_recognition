import cv2  
import numpy as np


def color_segmentation (img:int,limit:int):
    full_mask=0                                         # Задаем переменную для будующей маски 
    hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)        # Создаем копию изображения в формате HSV
    for i in range (len(limit))[::2]:                   # Проходимся по списку ограничений 
                                                        #(выбираем каждый второй элемент тк параметры для цвета идут парой)
        full_mask+=cv2.inRange(hsv_img, limit[i], limit[i+1])   # Создаем Маску по цвету и объединяем в общую
    final_result = cv2.bitwise_and(img, img, mask=full_mask)    # Создаем изображение с наложенной маской 
    return final_result