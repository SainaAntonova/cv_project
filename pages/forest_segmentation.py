import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import time
from PIL import Image
from torchvision import transforms as T
import numpy as np
from matplotlib import pyplot as plt
import requests
import io
st.image("https://eos.com/wp-content/uploads/2021/06/item-3-min.jpg.webp", use_column_width=True)
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
st.markdown('<h1 class="title">🛰️📷 Сегментация спутниковых снимков леса </h1>', unsafe_allow_html=True)
st.write(""" Обучен на датасете Kaggle - [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation/data)""")
with st.expander("Информация об обучении"):
    st.write('## Информация об обучении')
    st.write('- Объем выборки: 5 108 изображений')
    


######################### НАЧАЛО ТЕРРИТОРИИ ИГОРЯ #########################################################
# Загрузка модели
class ConvNormAct(nn.Module):
    def __init__(self, in_nc, out_nc, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_nc, out_nc, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_nc)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        return out

class UnetBlock(nn.Module):
    def __init__(self, in_nc, inner_nc, out_nc, inner_block=None):
        super().__init__()

        self.conv1 = ConvNormAct(in_nc, inner_nc, stride=2)
        self.conv2 = ConvNormAct(inner_nc, inner_nc)
        self.inner_block = inner_block
        self.conv3 = ConvNormAct(inner_nc, inner_nc)
        self.conv_cat = nn.Conv2d(inner_nc + in_nc, out_nc, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.shape

        inner = self.conv1(x)
        inner = self.conv2(inner)
        if self.inner_block is not None:
            inner = self.inner_block(inner)
        inner = self.conv3(inner)

        inner = F.upsample(inner, size=(h, w), mode='bilinear')
        # Конкатенация результата внутреннего блока и результата верхнего
        inner = torch.cat((x, inner), axis=1)
        out = self.conv_cat(inner)

        return out

class Unet(nn.Module):
    def __init__(self, in_nc=1, nc=32, out_nc=1, num_downs=6):
        super().__init__()

        self.cna1 = ConvNormAct(in_nc, nc)
        self.cna2 = ConvNormAct(nc, nc)

        unet_block = None
        for i in range(num_downs - 3):
            unet_block = UnetBlock(8 * nc, 8 * nc, 8 * nc, unet_block)
        unet_block = UnetBlock(4 * nc, 8 * nc, 4 * nc, unet_block)
        unet_block = UnetBlock(2 * nc, 4 * nc, 2 * nc, unet_block)
        self.unet_block = UnetBlock(nc, 2 * nc, nc, unet_block)

        self.cna3 = ConvNormAct(nc, nc)

        self.conv_last = nn.Conv2d(nc, out_nc, 3, padding=1)

    def forward(self, x):
        out = self.cna1(x)
        out = self.cna2(out)
        out = self.unet_block(out)
        out = self.cna3(out)
        out = self.conv_last(out)

        return out

# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
unet_model = Unet(in_nc=3, nc=32, out_nc=1, num_downs=5)
unet_model = unet_model.to(device)

# Загрузка сохраненной модели
checkpoint_path = 'notebooks/model/unet_model/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
unet_model.load_state_dict(checkpoint)

resize = T.Resize((256, 256))

def apply_colormap(image):
    colormap = plt.get_cmap('viridis')
    image = colormap(image)[:, :, :3]  # Apply colormap and remove alpha channel
    return (image * 255).astype(np.uint8)  # Convert to 8-bit image

# def forest_segmentation():
#     # Загрузка изображения через Streamlit
#     # Путь к изображению
#     image_path = 'notebooks/model/unet_model'
#     with st.expander("Show model info"):
#         st.write('''
#             Training dataset size: 4086 \n
#             Validation dataset size: 1022
#         ''')
#         st.image(image_path+'Arch.png', caption="Unet info", use_column_width=True)
#         col3, col4 = st.columns(2)
#         with col3:
#             st.image(image_path+'Loss.png', caption="Loss", use_column_width=True)
#             st.image(image_path+'Recall.png', caption="Loss", use_column_width=True)
#         with col4:
#             st.image(image_path+'IoU.png', caption="IoU", use_column_width=True)
#             # st.image(image_path+'Precision.png', caption="Loss", use_column_width=True)

#     with st.sidebar:
#         st.title("Загрузка изображения")

#         upload_type = st.radio(
#             label="Как загрузить картинку?",
#             options=("Из файла", "Из URL"),
#         )

#         image = []
#         if upload_type == "Из файла":
#             files = st.file_uploader(
#                 "Загрузите свои картинки", type=["jpg", "jpeg", "png"], accept_multiple_files=True
#             )
#             if files:
#                 image = files

#         if upload_type == "Из URL":
#             url = st.text_input("Введите URL")
#             if url:
#                 response = requests.get(url)
#                 img_bytes = response.content
#                 img = Image.open(io.BytesIO(img_bytes))
#                 image.append(img)

#     if image:
#         start_time = time.time()
#         for images in image:
#             # Преобразование загруженного файла в PIL Image
#             if isinstance(images, (str, io.BytesIO)):
#                 image1 = Image.open(images)
#             else:
#                 image1 = images

#             original_size = image1.size
#             img_tensor = resize(T.ToTensor()(image1)).unsqueeze(0).to(device)

#             # Модель в режиме оценки
#             unet_model.eval()
#             with torch.no_grad():
#                 output = unet_model(img_tensor)
#                 output = F.sigmoid(output).cpu().numpy()[0, 0]  # Remove batch and channel dimensions
#                 output = Image.fromarray((output * 255).astype(np.uint8))
#                 output = output.resize(original_size, Image.BILINEAR)
#                 output = np.array(output)
#                 colored_output = apply_colormap(output / 255.0)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image1, caption="Uploaded Image", use_column_width=True)
#             with col2:
#                 st.image(colored_output, caption="Segmented Image", use_column_width=True)

#         end_time = time.time()
#         execution_time = end_time - start_time
#         st.write(f"Время выполнения программы: {round(execution_time, 4)} секунд")

######################### КОНЕЦ ТЕРРИТОРИИ ИГОРЯ #########################################################



# Загрузка изображения через Streamlit
image_path = 'notebooks/model/unet_model/'
with st.expander("Show model info"):
    st.write('''
        Training dataset size: 4086 \n
        Validation dataset size: 1022
    ''')
    st.image(image_path+'Arch.png', caption="Unet info", use_column_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.image(image_path+'Loss.png', caption="Loss", use_column_width=True)
        st.image(image_path+'Recall.png', caption="Loss", use_column_width=True)
    with col4:
        st.image(image_path+'IoU.png', caption="IoU", use_column_width=True)

with st.sidebar:
    st.title("Загрузка изображения")
    upload_type = st.radio("Как загрузить картинку?", options=("Из файла", "Из URL"))

    images = []
    if upload_type == "Из файла":
        files = st.file_uploader("Загрузите свои картинки", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files:
            images = files

    if upload_type == "Из URL":
        url = st.text_input("Введите URL")
        if url:
            response = requests.get(url)
            img_bytes = response.content
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

if images:
    start_time = time.time()
    for img in images:
        if isinstance(img, (str, io.BytesIO)):
            img = Image.open(img)
        original_size = img.size
        img_tensor = resize(T.ToTensor()(img)).unsqueeze(0).to(device)
        unet_model.eval()
        with torch.no_grad():
            output = unet_model(img_tensor)
            output = F.sigmoid(output).cpu().numpy()[0, 0]
            output = Image.fromarray((output * 255).astype(np.uint8))
            output = output.resize(original_size, Image.BILINEAR)
            output = np.array(output)
            colored_output = apply_colormap(output / 255.0)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(colored_output, caption="Segmented Image", use_column_width=True)

    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Время выполнения программы: {round(execution_time, 4)} секунд")


# def main():
#     page = st.sidebar.selectbox("Выберите страницу", ["Сегментация спутниковых снимков"])

#     if page == "Сегментация спутниковых снимков":
#         forest_segmentation()

# if __name__ == '__main__':
#     main()
