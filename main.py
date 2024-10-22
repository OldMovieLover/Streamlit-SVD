import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from skimage import io
import requests

# Заголовок приложения
st.title("Анализ изображений с использованием SVD")

# Описание приложения
message = (
    "## Добро пожаловать!\n\n"
    "#### Это симулятор злой мамы :)\n\n"
    "##### Узнайте как испортиться ваше зрение от злых компьютьерных игр!"
)
st.markdown(message)

# Функция для загрузки изображения с кэшированием
@st.cache_data
def load_image(image_source, is_url=False):
    try:
        if is_url:
            # Пытаемся загрузить изображение по URL
            response = requests.get(image_source)
            if response.status_code == 200:
                st.success("Данные файла успешно получены по URL!")
                image = io.imread(image_source)[:, :, 1]  # Используем только один канал
            else:
                st.error("Не удалось загрузить изображение с URL. Проверьте адрес.")
                return None
        else:
            # Загружаем локально загруженное изображение
            image = io.imread(image_source)[:, :, 1]  # Используем только один канал
            st.success("Файл успешно загружен!")
        return image
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения: {e}")
        return None


# Применяем SVD
@st.cache_data
def perform_SVD(image, top_k):
    U, sing_vals, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_vals)

    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]

    return trunc_U, trunc_sigma, trunc_V, sing_vals

# Ввод информации пользователем
image_URL = st.text_input("Введите URL, чтобы загрузить изображение!")
uploaded_image = st.file_uploader('Загрузите изображение', type=['png', 'jpg', 'jpeg'])

image = None
if uploaded_image:
    image = load_image(uploaded_image)
elif image_URL:
    image = load_image(image_URL, is_url=True)


# Проверка на наличие загруженного изображения
if image is not None:
    # Добавляем слайдер, максимальное значение которого равно ширине изображения
    max_components = image.shape[0]  # Ширина изображения
    top_k = st.slider(f"Сколько чаосв в месяц вы не проводите в играх?\n(кол-во компонентов)", min_value=1, max_value=max_components, value=10)

    # Выполняем SVD
    trunc_U, trunc_sigma, trunc_V, sing_vals = perform_SVD(image, top_k)
    
    # Вывод информации о сохраненной дисперсии
    st.write(f'Сохранено {100 * top_k / len(sing_vals):.2f}% информации.')

    # Визуализация изображений
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 20))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Исходное изображение')
    axes[0].axis('off')  # Отключаем оси для чистоты отображения

    axes[1].imshow(trunc_U @ trunc_sigma @ trunc_V, cmap='gray')
    axes[1].set_title(f'Как вы будете видеть этот мир! Если тратить по {max_components-top_k} часов в месяц!\nВосстановленное изображение с {top_k} компонентами')
    axes[1].axis('off')  # Отключаем оси для чистоты отображения

    # Отображение графика
    st.subheader("Результаты анализа изображения с помощью SVD")
    st.pyplot(fig)
else:
    st.warning("Пожалуйста, загрузите изображение или введите URL")
