## Для запуска обучения установить зависимости в чистое окружение:

## Затем в корневую папку переместить папку, которая содержит split.json, train/, val/

## Запуск обучения:

```bash
python3.10 train.py --data ./folder_name
```

## Запуск тестов в двух вариантах для удобства:
1. через test.py
```bash
python3.10 test.py --artifacts ./artifacts --source bottom --x_y 2612 1177
```
2. через interact_test.py с отображением картинок
```bash
python3.10 interact_test.py --image ./folder_with_data/train/camera_door2_2025-11-27_16-52-37/bottom/frame_000139.jpg
```