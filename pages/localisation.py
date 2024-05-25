import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
import cv2
import numpy as np
import time 

import torchvision.transforms as T

preprocessing_func = T.Compose(
    [
        T.Resize((227,227)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing_func(img)




class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)
        )

        self.box = nn.Sequential(
            nn.Linear(512*8*8, 128),
            nn.Sigmoid(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        resnet_out = self.feature_extractor(img)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pred_classes = self.clf(resnet_out)
        pred_boxes = self.box(resnet_out)
        return pred_classes, pred_boxes
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_data
def load_model():
    
    model = Classifier()

    checkpoint_path = 'notebooks/model/loc_model/best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


model = load_model()

st.image("https://previews.123rf.com/images/foodandmore/foodandmore1705/foodandmore170500090/77253581-panoramic-wide-organic-food-background-concept-with-full-frame-pile-of-fresh-vegetables-and-fruits.jpg", use_column_width=True)
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
st.markdown('<h1 class="title">📷 Локализация объектов на основе ResNet-18</h1>', unsafe_allow_html=True)
st.write("""
         Этот веб-интерфейс представляет собой инструмент для локализации объектов на изображении с использованием модели, основанной на архитектуре ResNet-18.
         Модель классифицирует и локализует объекты на изображении, отображая прямоугольные рамки вокруг обнаруженных объектов и соответствующие метки классов.
    """)

with st.expander("Информация об обучении"):
    st.write('## Информация об обучении')
    st.write('- Число классов: 3 - огурец, баклажан, гриб')
    st.write('- Объем выборки: 186 изображений')

with st.expander("Метрики"):
    st.write('## Метрики:')
    st.write('  - mAP50: ???  - mAP50-95: ???')


label_dict = {
            'cucumber' : 0,
            'eggplant' : 1,
            'mushroom' : 2
            }

ix2cls = {v: k for k, v in label_dict.items()}



def predict(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        start_time = time.time()
        preds_cls, preds_reg = model(img)
        end_time = time.time()

    pred_class = preds_cls.argmax(dim=1).item()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)  # Преобразование значений пикселей в диапазон [0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Переключаем обратно в BGR для OpenCV

    pred_box_coords = (preds_reg[0] * 227).cpu().detach().numpy().astype('int')
    pred_box = cv2.rectangle(
        img.copy(), 
        (pred_box_coords[0], pred_box_coords[1]),  # top left
        (pred_box_coords[2], pred_box_coords[3]),  # bottom right
        color=(255, 0, 0), thickness=2
    )

    # Преобразование изображения обратно в RGB для правильного отображения в Streamlit
    pred_box = cv2.cvtColor(pred_box, cv2.COLOR_BGR2RGB)
    pred_box = pred_box / 255.0  # Нормализация значений пикселей в диапазон [0, 1]

    return pred_box, ix2cls[pred_class], end_time - start_time




st.write('## Ваше изображение:')

with st.sidebar:
    uploaded_files = st.file_uploader('Загрузите свою картинку', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

def display_results(image, pred_label, inference_time):
    st.title(pred_label)
    st.image(image, use_column_width=True)
    st.write(f"Время предсказания: {inference_time:.4f} секунд")


if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        predicted_img, predicted_class, inference_time = predict(image)
        display_results(predicted_img, predicted_class, inference_time)
