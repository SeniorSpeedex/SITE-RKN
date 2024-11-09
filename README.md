# Сайт по поиску смысловых копий изображений для ФГУП "ГРЧЦ"
**Структура проекта для запуска:**
   /Наш проект/
      - ├── app2.py
     -  ├── templates/
     -  │   └── index2.html
     -  └── static/
      -     ├── images/
         -  ├── style2.css
          - └── logo.png 
   ```bash
   python app2.py
   ```
**Применяемые технологии:**
1) Flask(Последняя версия)
2) NumPy(Последняя версия)
3) TensorFlow.keras(Последняя версия)
4) Модель VGG16(Включена в Keras)
5) Zipfile, pandas - для формирования ключевых наборов работы с моделью
6) PIL - для работы с изображениями
   -Всё является Open-Source и последними версиями!
```bash
pip install tensorflow numpy pandas flask zipfile pillow mimetypes datetime
```
**Kill-фичи:**
1) Возможность скачивания изображений
2) История поиска изображений
3) Реализация системы фидбеков

**TO-DO лист:**
1) Расширить список определяемого программой контента до видеороликов
2) Дообучить модель на фотографиях с экстримистскими наклонностями для 
блокировки подобного контента автоматически
3) Использовать фидбеки для дообучения модели.

# - Перед использованием пополните папка static/images изображениями для модели!!
