import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Папка, в якій містяться зображення
image_folder = "C:\\Users\\Yana\\Desktop\\CNN\\cell_images"

# Завантаження попередньо навченої моделі
model = tf.keras.models.load_model("C:\\Users\\Yana\\Desktop\\CNN\\malaria_detection_model.h5")

# Функція для завантаження та підготовки даних
def load_data(image_folder):
    # Список файлів зображень та їх міток
    images = []
    labels = []

    # Проходження через всі файли в папці
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image) / 255.0
        images.append(image)
        labels.append(1)  # Тут можна встановити відповідну мітку

    # Перетворення списків у масиви
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Загальні параметри
img_width = 80
img_height = 80

# Завантаження та підготовка даних
images, labels = load_data(image_folder)

# Завантаження нового набору даних змішаних інфікованих та неінфікованих фото
new_image_folder = "C:\\Users\\Yana\\Desktop\\CNN\\cell_images1"
new_images, new_labels = load_data(new_image_folder)

# Аналіз нового набору даних за допомогою попередньо навченої нейронної мережі
predictions = model.predict(new_images)

# Виведення списку інфікованих фото
infected_filenames = [filename for filename, prediction in zip(os.listdir(new_image_folder), predictions) if prediction > 0.5]
print("Список інфікованих фото:")
for filename in infected_filenames:
    print(filename)


