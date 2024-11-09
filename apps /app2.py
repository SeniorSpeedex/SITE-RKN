import os
import numpy as np
import mimetypes
from flask import Flask, request, render_template, session, send_file
from datetime import datetime
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import concurrent.futures
from PIL import Image
import pandas as pd
import zipfile

app = Flask(__name__)
app.secret_key = 'your_secret_key'

IMAGE_FOLDER = 'static/images/'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

model = applications.VGG16(weights='imagenet', include_top=False, pooling='avg')


def is_image(file):
    mime = mimetypes.guess_type(file)[0]
    return mime and mime.startswith('image')


def preprocess_image(img_path):
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
    img_array = preprocess_image(img_path)
    if img_array is None:
        raise ValueError("Не удалось извлечь признаки изображения.")
    features = model.predict(img_array).flatten()
    return features


def similarity_score(feat1, feat2):
    dot_product = np.dot(feat1, feat2)
    norm_a = np.linalg.norm(feat1)
    norm_b = np.linalg.norm(feat2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def process_image(img):
    img_path = os.path.join(IMAGE_FOLDER, img)
    img_features = extract_features(img_path)
    return img, img_features


def generate_recommendations(image_name):
    return [f"{image_name}_recomm_{i}" for i in range(1, 11)]


def save_recommendations_to_csv(image_name, recommendations):
    response_data = {
        'image': image_name,
        'recs': '"' + ', '.join(recommendations) + '"'
    }

    df = pd.DataFrame([response_data])
    df.to_csv('submission.csv', index=False, quoting=1)


@app.route('/')
def index():
    return render_template('index2.html', results=None, history=[], feedback_status=None, uploaded_image=None)


@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image' not in request.files:
        return "Нет изображения для загрузки", 400

    image_file = request.files['image']

    if not (image_file.filename.endswith('.png') or image_file.filename.endswith(
            '.jpg') or image_file.filename.endswith('.jpeg')):
        return "Пожалуйста, загрузите изображение в формате PNG, JPG или JPEG.", 400

    original_image_path = os.path.join(IMAGE_FOLDER, image_file.filename)
    image_file.save(original_image_path)

    print(f"Файл сохранен: {original_image_path}, размер: {os.path.getsize(original_image_path)} байт")

    original_features = extract_features(original_image_path)

    all_images = [f for f in os.listdir(IMAGE_FOLDER) if f != image_file.filename]
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_to_img = {executor.submit(process_image, img): img for img in all_images}
        for future in concurrent.futures.as_completed(future_to_img):
            try:
                img, img_features = future.result()
                sim_score = similarity_score(original_features, img_features)

                if sim_score > 0.53:
                    results.append(img)

                if len(results) >= 20:
                    break
            except Exception as e:
                print(f"Ошибка обработки изображения {future_to_img[future]}: {e}")

    status = "полный" if len(results) >= 10 else "неполный"

    history_entry = {
        'date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'similarity': np.mean(
            [similarity_score(original_features, extract_features(os.path.join(IMAGE_FOLDER, img))) for img in results]
        ) if results else 0,
        'images_found': ', '.join(results)
    }

    with open('search_history.csv', 'a') as f:
        f.write(','.join(str(value) for value in history_entry.values()) + '\n')

    session['results'] = results
    session['uploaded_image'] = image_file.filename

    # Генерация рекомендаций и сохранение CSV
    recommendations = generate_recommendations(image_file.filename.split('.')[0])
    save_recommendations_to_csv(image_file.filename.split('.')[0], recommendations)

    return render_template('index2.html', uploaded_image=image_file.filename, results=results)


@app.route('/history', methods=['GET'])
def view_history():
    """Отображение истории поиска."""
    if not os.path.exists('search_history.csv'):
        history_data = []
    else:
        # Чтение данных из CSV с указанием кодировки и обработкой плохих строк
        history_data = pd.read_csv(
            'search_history.csv',
            header=None,
            names=['date_time', 'status', 'similarity', 'images_found'],
            encoding='latin1',
            on_bad_lines='warn'  # Предупреждать о плохих строках
        ).values.tolist()

    return render_template('history.html', history=history_data)




@app.route('/download', methods=['GET'])
def download_zip():
    """Создание ZIP-архива с найденными изображениями и его скачивание."""
    if 'results' not in session:
        return "Нет изображений для скачивания", 400

    zip_file_path = 'found_images.zip'

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for img in session['results']:
            img_path = os.path.join(IMAGE_FOLDER, img)
            zipf.write(img_path, os.path.basename(img_path))

    return send_file(zip_file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
