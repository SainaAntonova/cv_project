# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import streamlit as st
# import time
# from PIL import Image
# from torchvision import transforms as T
# from torchvision import io
# from matplotlib import pyplot as plt
#
#
# ######################### НАЧАЛО ТЕРРИТОРИИ ИГОРЯ #########################################################
# # Загрузка модели
# class ConvNormAct(nn.Module):
#     def __init__(self, in_nc, out_nc, stride=1):
#         super().__init__()
#
#         self.conv = nn.Conv2d(in_nc, out_nc, 3, stride=stride, padding=1, bias=False)
#         self.norm = nn.BatchNorm2d(out_nc)
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.act(out)
#
#         return out
#
# class UnetBlock(nn.Module):
#     def __init__(self, in_nc, inner_nc, out_nc, inner_block=None):
#         super().__init__()
#
#         self.conv1 = ConvNormAct(in_nc, inner_nc, stride=2)
#         self.conv2 = ConvNormAct(inner_nc, inner_nc)
#         self.inner_block = inner_block
#         self.conv3 = ConvNormAct(inner_nc, inner_nc)
#         self.conv_cat = nn.Conv2d(inner_nc+in_nc, out_nc, 3, padding=1)
#
#     def forward(self, x):
#         _,_,h,w = x.shape
#
#         inner = self.conv1(x)
#         inner = self.conv2(inner)
#         if self.inner_block is not None:
#             inner = self.inner_block(inner)
#         inner = self.conv3(inner)
#
#         inner = F.upsample(inner, size=(h,w), mode='bilinear')
#         # Конкатенация результата внутреннего блока и результата верхнего
#         inner = torch.cat((x, inner), axis=1)
#         out = self.conv_cat(inner)
#
#         return out
#
# class Unet(nn.Module):
#     def __init__(self, in_nc=1, nc=32, out_nc=1, num_downs=6):
#         super().__init__()
#
#         self.cna1 = ConvNormAct(in_nc, nc)
#         self.cna2 = ConvNormAct(nc, nc)
#
#         unet_block = None
#         for i in range(num_downs-3):
#             unet_block = UnetBlock(8*nc, 8*nc, 8*nc, unet_block)
#         unet_block = UnetBlock(4*nc, 8*nc, 4*nc, unet_block)
#         unet_block = UnetBlock(2*nc, 4*nc, 2*nc, unet_block)
#         self.unet_block = UnetBlock(nc, 2*nc, nc, unet_block)
#
#         self.cna3 = ConvNormAct(nc, nc)
#
#         self.conv_last = nn.Conv2d(nc, out_nc, 3, padding=1)
#
#     def forward(self, x):
#         out = self.cna1(x)
#         out = self.cna2(out)
#         out = self.unet_block(out)
#         out = self.cna3(out)
#         out = self.conv_last(out)
#
#         return out
#
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
#
# unet_model = Unet(in_nc=3, nc=32, out_nc=1, num_downs=5)
# unet_model = unet_model.to(device)
#
# # Загрузка сохраненной модели
# checkpoint_path = 'C:/Users/igors/forest_project/forest_segmentation_model.pth'
# checkpoint = torch.load(checkpoint_path)
# unet_model.load_state_dict(checkpoint)
#
# resize = T.Resize((256, 256))
#
# def forest_segmentation():
#     # Загрузка изображения через Streamlit
#     uploaded_files = st.file_uploader(label='Upload jpg file', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
#
#     if uploaded_files:
#         start_time = time.time()
#         for uploaded_file in uploaded_files:
#             # Преобразование загруженного файла в PIL Image
#             image = Image.open(uploaded_file)
#             img_tensor = resize(T.ToTensor()(image)).unsqueeze(0).to(device)
#
#             # Отображение изображения
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Модель в режиме оценки
#             unet_model.eval()
#             with torch.no_grad():
#                 output = unet_model(img_tensor)
#                 output = F.sigmoid(output).cpu().numpy()[0].transpose(1, 2, 0)
#                 st.image(output, caption="Segmented Image", use_column_width=True)
#
#         end_time = time.time()
#         execution_time = end_time - start_time
#         st.write(f"Время выполнения программы: {round(execution_time, 4)} секунд")
#
# ######################### КОНЕЦ ТЕРРИТОРИИ ИГОРЯ #########################################################
#
# def main():
#     page = st.sidebar.selectbox("Выберите страницу", ["Сегментация спутниковых снимков"])
#
#     if page == "Сегментация спутниковых снимков":
#         forest_segmentation()
#
# if __name__ == '__main__':
#     main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import time
from PIL import Image
from torchvision import transforms as T
from torchvision import io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


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
        self.conv_cat = nn.Conv2d(inner_nc+in_nc, out_nc, 3, padding=1)

    def forward(self, x):
        _,_,h,w = x.shape

        inner = self.conv1(x)
        inner = self.conv2(inner)
        if self.inner_block is not None:
            inner = self.inner_block(inner)
        inner = self.conv3(inner)

        inner = F.upsample(inner, size=(h,w), mode='bilinear')
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
        for i in range(num_downs-3):
            unet_block = UnetBlock(8*nc, 8*nc, 8*nc, unet_block)
        unet_block = UnetBlock(4*nc, 8*nc, 4*nc, unet_block)
        unet_block = UnetBlock(2*nc, 4*nc, 2*nc, unet_block)
        self.unet_block = UnetBlock(nc, 2*nc, nc, unet_block)

        self.cna3 = ConvNormAct(nc, nc)

        self.conv_last = nn.Conv2d(nc, out_nc, 3, padding=1)

    def forward(self, x):
        out = self.cna1(x)
        out = self.cna2(out)
        out = self.unet_block(out)
        out = self.cna3(out)
        out = self.conv_last(out)

        return out

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

unet_model = Unet(in_nc=3, nc=32, out_nc=1, num_downs=5)
unet_model = unet_model.to(device)

# Загрузка сохраненной модели
checkpoint_path = 'C:/Users/igors/forest_project/best_model.pth'
checkpoint = torch.load(checkpoint_path)
unet_model.load_state_dict(checkpoint)

resize = T.Resize((256, 256))

def apply_colormap(image):
    colormap = plt.get_cmap('viridis')
    image = colormap(image)[:, :, :3]  # Apply colormap and remove alpha channel
    return (image * 255).astype(np.uint8)  # Convert to 8-bit image

def forest_segmentation():
    # Загрузка изображения через Streamlit
    uploaded_files = st.file_uploader(label='Upload jpg file', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        start_time = time.time()
        for uploaded_file in uploaded_files:
            # Преобразование загруженного файла в PIL Image
            image = Image.open(uploaded_file)
            img_tensor = resize(T.ToTensor()(image)).unsqueeze(0).to(device)

            # Отображение изображения
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Модель в режиме оценки
            unet_model.eval()
            with torch.no_grad():
                output = unet_model(img_tensor)
                output = F.sigmoid(output).cpu().numpy()[0, 0]  # Remove batch and channel dimensions
                colored_output = apply_colormap(output)

                st.image(colored_output, caption="Segmented Image", use_column_width=True)

        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Время выполнения программы: {round(execution_time, 4)} секунд")

######################### КОНЕЦ ТЕРРИТОРИИ ИГОРЯ #########################################################

def main():
    page = st.sidebar.selectbox("Выберите страницу", ["Сегментация спутниковых снимков"])

    if page == "Сегментация спутниковых снимков":
        forest_segmentation()

if __name__ == '__main__':
    main()
