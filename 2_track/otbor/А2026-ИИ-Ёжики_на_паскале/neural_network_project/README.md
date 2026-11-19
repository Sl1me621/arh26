# Underwater Black Box Digit Classifier

Проект обучает и сравнивает две модели для 10 классов:
`digit_1`...`digit_9` и `no_object`.

Основной сценарий:

- MobileNetV2 загружается из `models/mobilenetv2_best.h5`, если файл уже есть,
  затем дообучается через fine-tuning последних слоев backbone.
- MobileNetV3Small обучается в два этапа: сначала замороженный backbone и
  классификационная голова, затем fine-tuning последних слоев backbone.
- Финальная модель выбирается по `f1_macro` на test split. Если разница
  `f1_macro` меньше `0.02`, выбирается более быстрая модель; при похожей
  скорости выбирается меньшая модель.

Resize до `224x224` выполняется только в preprocessing pipeline. Исходные
изображения не перезаписываются. Horizontal flip не используется.

## Установка

Из папки `neural_network_project`:

```bash
python -m pip install -r requirements.txt
```

## Обучение, fine-tuning, сравнение и экспорт

```bash
python train_compare.py --epochs 20 --batch-size 32
```

Эквивалентно с явным указанием этапов:

```bash
python train_compare.py --head-epochs 20 --finetune-epochs 12 --batch-size 32 --finetune-learning-rate 1e-5 --finetune-layers 40
```

Если часть моделей уже есть и нужно продолжить без переобучения готовых
финальных checkpoint-файлов:

```bash
python train_compare.py --epochs 20 --batch-size 32 --resume-existing
```

Скрипт проверяет:

- наличие `../dataset_split/train`, `../dataset_split/val`, `../dataset_split/test`;
- наличие ровно 10 классов;
- читаемость изображений;
- исходное разрешение не меньше `640x480`.

Аугментации безопасные для цифр: небольшой rotation, zoom, brightness и
contrast. Horizontal flip не применяется.

## Сохранённые результаты

`model.h5` побайтово совпадает с `models/mobilenetv2_finetuned_best.h5`. Также включён `models/mobilenetv3small_best.h5`. Для MobileNetV2 доступны [метрики и таблица ошибок](metrics/) и [графики](plots/). В отчёте тестовой выборки из 1 935 изображений accuracy равна 0,9587, macro-F1 — 0,9553.

Скрипт полного эксперимента предусматривает сравнение моделей и экспорт в SavedModel и TFLite. Эти экспортные файлы, результаты сравнения с MobileNetV3Small и исходные выборки в данной ветке отсутствуют.

## Предсказание

```bash
python predict.py --model model.h5 --image path/to/image.jpg
```

Или для конкретного checkpoint:

```bash
python predict.py --model models/mobilenetv2_finetuned_best.h5 --image path/to/image.jpg
```

Скрипт выводит predicted class, confidence и top-3 predictions.

## Dataset.zip для сдачи

Если нужен архив датасета в регламентной структуре:

```bash
python prepare_submission_dataset.py
```

Будет создан `../dataset.zip` со структурой:

```text
dataset/
  black_box/
  no_object/
```

Файлы из `dataset/black_box/digit_1`...`digit_9` собираются в одну папку
`dataset/black_box/` внутри архива с уникальными `.jpg` именами. Рабочая
структура датасета не меняется.
