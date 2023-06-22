import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


def evaluate_model(image_path):
    # Шлях до файлу з моделлю
    model_path = "C:\\Users\\Yana\\Desktop\\CNN\\malaria_detection_model.h5"

    # Завантаження моделі
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        messagebox.showerror("Помилка", "Модель не знайдена.")
        return

    # Загальні параметри
    img_width = 80
    img_height = 80

    # Завантаження та підготовка зображення
    image = Image.open(image_path)
    image = image.resize((img_width, img_height))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Оцінка моделі
    prediction = model.predict(image)
    if prediction > 0.5:
        result = "Інфіковано"
    else:
        result = "Неінфіковано"
    messagebox.showinfo("Результат", result)


def diagnose_blood_samples():
    # Папка, в якій містяться зображення
    image_folder = "C:\\Users\\Yana\\Desktop\\CNN\\cell_images1"

    # Шлях до файлу з моделлю
    model_path = "C:\\Users\\Yana\\Desktop\\CNN\\malaria_detection_model.h5"

    # Завантаження моделі
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        messagebox.showerror("Помилка", "Модель не знайдена.")
        return

    # Загальні параметри
    img_width = 80
    img_height = 80

    # Вибір зображення для оцінки
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title='Виберіть зображення')

    # Перевірка, чи вибрано зображення
    if not image_path:
        return

    # Виклик функції для оцінки моделі з передачею шляху до зображення
    evaluate_model(image_path)


def show_infected_list():
    root = tk.Tk()
    root.withdraw()
    # Папка, в якій містяться зображення
    image_folder = "C:\\Users\\Yana\\Desktop\\CNN\\cell_images1"

    # Шлях до файлу з моделлю
    model_path = "C:\\Users\\Yana\\Desktop\\CNN\\malaria_detection_model.h5"

    # Завантаження моделі
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        messagebox.showerror("Помилка", "Модель не знайдена.")
        return

    # Загальні параметри
    img_width = 80
    img_height = 80

    # Завантаження та підготовка даних
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images = []
    labels = []
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image) / 255.0
        images.append(image)
        labels.append(1)  # Тут можна встановити відповідну мітку
    images = np.array(images)
    labels = np.array(labels)

    # Аналіз нового набору даних за допомогою попередньо навченої нейронної мережі
    predictions = model.predict(images)

    # Створення вікна для виведення списку інфікованих фото з прокручуванням
    window = tk.Tk()
    window.title("Список інфікованих фото")

    # Створення області прокручування
    scrollbar = tk.Scrollbar(window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Створення текстового поля для виведення списку
    infected_list = tk.Text(window, yscrollcommand=scrollbar.set)
    infected_list.pack(side=tk.LEFT, fill=tk.BOTH)

    # Додавання інфікованих фото в текстове поле
    for filename, prediction in zip(image_files, predictions):
        if prediction > 0.5:
            infected_list.insert(tk.END, filename + "\n")

    # Налаштування прокручування
    scrollbar.config(command=infected_list.yview)

    # Запуск графічного інтерфейсу
    window.mainloop()


def main():
    root = tk.Tk()
    root.title("Система діагностики малярії")
    root.geometry("400x300")
    root.configure(bg="DodgerBlue")

    # Функція для кнопки "Діагностика зразків крові"
    def evaluate_quality():
        image_path = filedialog.askopenfilename(title='Виберіть зображення')
        if image_path:
            evaluate_model(image_path)
            # Відображення зображення в графічному інтерфейсі
            image = Image.open(image_path)
            image = image.resize((200, 200))
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo  # Збереження посилання на зображення

    def show_infected_photos():
        show_infected_list()


    evaluate_button = tk.Button(root, text="Діагностика зразків крові", command=evaluate_quality, bg="SteelBlue", fg="LightCyan")
    evaluate_button.pack()

    infected_photos_button = tk.Button(root, text="Список інфікованих фото", command=show_infected_photos, bg="SteelBlue", fg="LightCyan")
    infected_photos_button.pack()

    # Відображення зображення
    image_label = tk.Label(root)
    image_label.pack()

    root.mainloop()


if __name__ == '__main__':
    main()








