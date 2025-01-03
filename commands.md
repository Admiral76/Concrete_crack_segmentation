### Билд образа
docker build -t concrete_crack_segmentation .

### Запуск тренировки внутри контейнера
docker run --rm -v data:/opt/pysetup/data -v models:/opt/pysetup/models concrete_crack_segmentation train.py

### Запуск инференса внутри контейнера
docker run --rm -v data:/opt/pysetup/data -v models:/opt/pysetup/models concrete_crack_segmentation infer.py
