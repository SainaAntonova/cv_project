import streamlit as st
import cv2
import numpy as np 
from PIL import Image
from io import BytesIO
import requests
from ultralytics import YOLO
import os
import torch
from torchvision import transforms

st.image("https://blogs.mathworks.com/deep-learning/files/2022/02/Cover.png", use_column_width=True)
st.markdown("""
    <style>
        .title {
            font-family: 'Arial Black', sans-serif;
            font-size: 36px;
            color: #1E90FF;
            text-align: center;
            margin-bottom: 30px;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        .stButton>button {
            color: white;
            background: linear-gradient(to right, #1fa2ff, #12d8fa, #a6ffcb);
        }
        .uploaded-image {
            border: 5px solid #1E90FF;
            border-radius: 10px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

# Отображение заголовка
st.markdown('<h1 class="title">⛴️📡📷 YOLOv8🛰️🛳️</h1>', unsafe_allow_html=True)
st.write("""
         
    Модель YOLOv8 была обучена на датасете [KAGGLE - Ships/Vessels in Aerial Images](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images/code)
    """)

with st.expander("Информация об обучении"):
    st.write('## Информация об обучении')
    st.write('- Число классов: 1 - корабль')
    st.write('- Объем выборки: 13 435 изображений')
    
    st.image('notebooks/model/yolo_runs/detect/train2/val_batch1_labels.jpg')
with st.expander("Метрики"):
    st.write('## Метрики:')
    st.write('  - mAP50: 0.5683,\n  - mAP50-95: 0.30734')
    st.image('notebooks/model/yolo_runs/detect/train2/results.png')
    st.write('  - График Percision-Recall кривой')
    st.image('notebooks/model/yolo_runs/detect/train2/PR_curve.png')
    st.write('  - Confusion Matrix')
    st.image('notebooks/model/yolo_runs/detect/train2/confusion_matrix_normalized.png')
    st.write('F1-кривая')
    st.image('notebooks/model/yolo_runs/detect/train2/F1_curve.png')



with st.sidebar:
    st.title("Загрузка изображения")

    upload_type = st.radio(
        label="Как загрузить картинку?",
        options=(("Из файла", "Из URL", "Из вебкамеры")),
    )

    image = None
    if upload_type == "Из файла":
        files = st.file_uploader(
            "Загрузите из папки или перетащите свои картинки", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        if files:
            image = [file.getvalue() for file in files]

    if upload_type == "Из URL":
        url = st.text_input("Введите URL")
        if url:
            image = [requests.get(url).content]

    if upload_type == "Из вебкамеры":
        camera = st.camera_input("### 🧀🧀🧀Сыыр!🧀🧀🧀")
        if camera:
            image = [camera.getvalue()]

st.write("## Загруженная картинка & Прогноз модели")
if image:
    st.write("🎉 Вот результат!")
    # st.image(image, width=200)
else:
    
    st.warning("👈 Пожалуйста, сначала загрузите изображение...")
    st.stop()



@st.cache_resource()
# YOLOv8
def load_model(weights_path='notebooks/model/yolo_runs/detect/train2/weights/best.pt'):
    model = YOLO(weights_path)
    return model

# # YOLOv5
# def load_model(weights_path='notebooks/model/yolo_runs/train/exp/weights/best.pt'):
#     model = YOLO(weights_path)
#     return model


# #пробный запуск
# def load_model():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Загрузка YOLOv5s
#     return model

model = load_model()

def predict(img):
    img = np.array(img)
    result = model(img)
    return result


# image = st.file_uploader("Загрузите своё изображение и увидите результаты детекции кораблей!", type=["jpg", "jpeg", "png"])

def load_image(image_file):
    if isinstance(image_file, str):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return image



if image:
    images = [Image.open(BytesIO(image)) for image in image]
    predictions = [predict(image) for image in images]
    # predictions = [predict(np.array(image)) for image in images]
    
    for image, prediction in zip(images, predictions):
        col1, col2= st.columns(2)
        with col1:
            # Отображение оригинального изображения
            st.image(image, caption='Оригинальное изображение', use_column_width=True)
        with col2:
            # Отображение изображения с детекцией объектов
            detected_image = prediction[0].plot()  # Получаем изображение с детекцией объектов
            st.image(detected_image, caption='Обнаруженные объекты', use_column_width=True)

    st.image("https://media0.giphy.com/media/9xnNG7EN2h822ithtT/giphy.gif?cid=6c09b952jkeqsu1vzio1sr6gr67m941155ubajei5yltumvk&ep=v1_gifs_search&rid=giphy.gif&ct=g", use_column_width=True)