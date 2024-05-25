import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from notebooks.model.loc_model.preprocessing import preprocess
import cv2
import numpy as np

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.clf = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.Tanh(),
            nn.Linear(256, 3)
        )

        self.box = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, img):
        embedding = self.feature_extractor(img)
        logits = self.clf(torch.flatten(embedding, 1))
        box_coords = self.box(torch.flatten(embedding, 1))
        return logits, box_coords

@st.cache_data
def load_model():
    model = Classifier()
    # Загрузка состояния модели и оптимизатора
    url = 'https://drive.google.com/file/d/1Zv6ojBq4jFyXE0AKVvYb5RifEmQzRBVN/view'
    output = 'best.pth'  # Local filename to save the downloaded model
    gdown.download(url, output, quiet=False)  # Download the model
    checkpoint = torch.load(output) 

    
    # checkpoint = torch.load('notebooks/model/loc_model/best.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25])

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    log = checkpoint['log']
    model.eval()
    return model, start_epoch

model, start_epoch = load_model()

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



with st.sidebar:
    image = st.file_uploader('Загрузите свою картинку', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

def predict(model, image, device='cpu'):
    img = preprocess(image)

    # Ensure img is a 4D tensor
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        img = img.to(device)
        logits, coords = model(img)
        logits = logits.cpu()
        coords = coords.cpu()
    return logits, coords

label_dict = {
            'cucumber' : 0,
            'eggplant' : 1,
            'mushroom' : 2
            }

ix2cls = {v: k for k, v in label_dict.items()}

st.write('## Ваше изображение:')
if image:
    for uploaded_image in image:
        image = Image.open(uploaded_image)
        orig_width, orig_height = image.size
        logits, coords = predict(model, image)
        # Масштабирование координат предсказанного бокса
        coords = coords.squeeze().numpy()
        xmin, ymin, xmax, ymax = coords
        pred_label = logits.argmax(1).item()
        pred_class = ix2cls[pred_label]
        image_np = np.array(image)
        cv2.rectangle(image_np, (int(xmin*orig_width), int(ymin*orig_height)), (int(xmax*orig_width), int(ymax*orig_height)), (255, 0, 0), 1)
        cv2.putText(image_np, pred_class, (int(xmin*orig_width), int(ymin*orig_height)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image = Image.fromarray(image_np)
        st.image(image)
        st.write(f'Количество эпох: {start_epoch}')
