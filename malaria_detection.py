import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py

# Папка, в якій містяться зображення
image_folder = "C:\\Users\\Yana\\Desktop\\CNN\\cell_images"

# Шлях до файлу, у якому буде збережена модель
model_path = "C:\\Users\\Yana\\Desktop\\CNN\\malaria_detection_model.h5"

# Завантаження та підготовка даних
def load_data(image_folder):
    # Список файлів зображень та їх міток
    images = []
    labels = []

    # Проходження через папку "Parasitized"
    parasitized_folder = os.path.join(image_folder, "Parasitized")
    parasitized_files = [f for f in os.listdir(parasitized_folder) if f.endswith('.png')]
    for filename in parasitized_files:
        image_path = os.path.join(parasitized_folder, filename)
        image = Image.open(image_path)
        image = image.resize((img_width, img_height))
        image = np.array(image) / 255.0
        images.append(image)
        labels.append(1)  # 1 - мітка для "Parasitized"

    # Проходження через папку "Uninfected"
    uninfected_folder = os.path.join(image_folder, "Uninfected")
    uninfected_files = [f for f in os.listdir(uninfected_folder) if f.endswith('.png')]
    for filename in uninfected_files:
        image_path = os.path.join(uninfected_folder, filename)
        image = Image.open(image_path)
        image = image.resize((img_width, img_height))
        image = np.array(image) / 255.0
        images.append(image)
        labels.append(0)  # 0 - мітка для "Uninfected"

    # Перетворення списків у масиви
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Загальні параметри
img_width = 80
img_height = 80

# Визначення функції split_data
def split_data(images, labels, test_size):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    return train_images, train_labels, test_images, test_labels

# Завантаження та підготовка даних
images, labels = load_data(image_folder)

# Розбиття даних на тренувальний та тестовий набори
train_images, train_labels, test_images, test_labels = split_data(images, labels, test_size=0.2)

# Перевірка наявності моделі
if os.path.isfile(model_path):
    # Завантаження збереженої моделі
    model = tf.keras.models.load_model(model_path)
else:
    # Створення нової моделі
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Компіляція моделі
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Навчання моделі
    model.fit(train_images, train_labels, epochs=5, batch_size=32)

    # Збереження моделі у форматі HDF5
    model.save(model_path)

# Оцінка моделі на тестових даних
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Прогнозування на тестових даних
y_pred = model.predict(test_images).ravel()

# Обчислення ROC-кривої
fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
roc_auc = auc(fpr, tpr)

# Побудова ROC-кривої
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



