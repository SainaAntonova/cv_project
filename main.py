import streamlit as st

st.set_page_config(layout="wide")


st.markdown("""
    <style>
        .title {
            font-family: 'Arial Black', sans-serif;
            font-size: 36px;
            color: #1E90FF;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

# Отображение заголовка
st.image("https://static.vecteezy.com/system/resources/thumbnails/021/938/316/small_2x/artificial-intelligence-icon-element-design-illustration-ai-technology-and-cyber-icon-element-futuristic-technology-service-and-communication-artificial-intelligence-concept-free-png.png", use_column_width=True)
st.markdown('<h1 class="title">📷Computer vision project • Mask-RCNN team </h1>', unsafe_allow_html=True)

st.write("""
         **Навигация по проекту:**
         - **[Локализация объектов](localisation):** В данном разделе реализована локализация объектов с использованием модели ResNet-18.
         - **[Детекция объектов](yolo_ships):** В этом разделе применена модель YOLOv8 для детекции кораблей на аэроснимках.
         - **[Семантическая сегментация](forest_segmentation):** Здесь используется модель Unet для сегментации объектов на аэроснимках.
         """)


st.write("""
         **Команда Mask-RCNN:**
         - **[Левон Мартиросян](https://github.com/kukerg33k?tab=repositories)**
         - **[Сайына Антонова](https://github.com/SainaAntonova)**
         - **[Игорь Свиланович](https://github.com/svilanovich)**
         """)

# from pages import yolo_ships, unet_segmentation, localisation
