import os
import numpy as np
import csv
from flask import Flask, request, render_template, session, send_file
from datetime import datetime
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import concurrent.futures
from PIL import Image
import zipfile

app = Flask(__name__)
app.secret_key = 'your_secret_key'

IMAGE_FOLDER = 'static/metric/'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Загружаем модель VGG16
model = applications.VGG16(weights='imagenet', include_top=False, pooling='avg')


def preprocess_image(img_path):
    """Преобразует изображение в формат, пригодный для модели."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        return None


def extract_features(img_path):
    """Извлекает признаки из изображения с использованием модели VGG16."""
    img_array = preprocess_image(img_path)
    if img_array is None:
        raise ValueError("Не удалось извлечь признаки изображения.")
    features = model.predict(img_array).flatten()
    return features


def similarity_score(feat1, feat2):
    """Вычисляет сходство между двумя наборами признаков."""
    dot_product = np.dot(feat1, feat2)
    norm_a = np.linalg.norm(feat1)
    norm_b = np.linalg.norm(feat2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def process_image(img):
    """Обрабатывает одно изображение для извлечения его признаков."""
    img_path = os.path.join(IMAGE_FOLDER, img)
    img_features = extract_features(img_path)
    return img, img_features


@app.route('/')
def index():
    """Отображает главную страницу."""
    return render_template('index2.html', results=None, history=[], feedback_status=None, uploaded_image=None)


@app.route('/compare', methods=['POST'])
def compare_images():
    """Сравнивает загруженное изображение с набором данных и генерирует рекомендации."""
    if 'image' not in request.files:
        return "Нет изображения для загрузки", 400

    image_file = request.files['image']

    if not (image_file.filename.endswith('.png') or image_file.filename.endswith('.jpg') or image_file.filename.endswith('.jpeg')):
        return "Пожалуйста, загрузите изображение в формате PNG, JPG или JPEG.", 400

    original_image_path = os.path.join(IMAGE_FOLDER, image_file.filename)
    image_file.save(original_image_path)

    print(f"Файл сохранен: {original_image_path}, размер: {os.path.getsize(original_image_path)} байт")

    original_features = extract_features(original_image_path)

    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f != image_file.filename]
    results = []

    # Порог похожести для изображений
    similarity_threshold = 0.5  # Установите порог

    # Асинхронная обработка изображений
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_img = {executor.submit(process_image, img): img for img in all_images}
        for future in concurrent.futures.as_completed(future_to_img):
            try:
                img, img_features = future.result()
                sim_score = similarity_score(original_features, img_features)

                if sim_score > similarity_threshold:
                    results.append(img)

                if len(results) >= 10:  # Если нашли 10 похожих изображений — выход
                    break
            except Exception as e:
                print(f"Ошибка обработки изображения {future_to_img[future]}: {e}")

    # Сообщение, если не удалось найти 10 изображений
    if len(results) < 10:
        if len(results) == 0:
            status = "Не найдено похожих изображений."
        else:
            status = f"Найдено только {len(results)} похожих изображений."
    else:
        status = "Полный список найденных изображений."

    # Сохраняем результаты в сессии
    session['found_images'] = results

    # Генерация файла submission.csv
    generate_submission_csv(image_file.filename, results)

    # Добавляем запись в историю
    if 'history' not in session:
        session['history'] = []
    history_entry = {
        'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'similarity': float(np.mean(
            [similarity_score(original_features, extract_features(os.path.join(IMAGE_FOLDER, img))) for img in
             results]) if results else 0),
        'images_found': ', '.join(results)
    }
    session['history'].append(history_entry)

    return render_template('index2.html', results=results, history=session['history'], feedback_status=None,
                           uploaded_image=image_file.filename)


def generate_submission_csv(uploaded_image, images):
    """Создаёт файл submission.csv с рекомендациями изображений."""
    filename = 'submission.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'recs'])

        # Убираем расширения и получаем первые 10 рекомендованных изображений
        recs = ','.join([os.path.splitext(img)[0] for img in images[:10]])

        # Записываем в файл
        writer.writerow([os.path.splitext(uploaded_image)[0], f'"{recs}"'])  # Включаем названия изображений в кавычках

    print(f"Создан файл {filename} с рекомендациями.")


@app.route('/history')
def history():
    """Отображает историю предыдущих сравнений."""
    return render_template('index2.html', results=None, history=session.get('history', []), feedback_status=None,
                           uploaded_image=None)


@app.route('/download', methods=['GET'])
def download():
    """Скачивает найденные изображения в zip-архиве."""
    zip_filename = 'similar_images.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img in session.get('found_images', []):
            img_path = os.path.join(IMAGE_FOLDER, img)
            zipf.write(img_path, arcname=img)

    return send_file(zip_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
