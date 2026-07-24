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

## Результаты

Модели:

- `models/mobilenetv2_finetuned_best.h5`
- `models/mobilenetv3small_best.h5`
- `models/mobilenetv3small_finetuned_best.h5`
- итоговая модель для сдачи: `model.h5`

Метрики:

- `metrics/mobilenetv2_finetuned_metrics.json`
- `metrics/mobilenetv3small_metrics.json`
- `metrics/comparison_table.csv`
- `metrics/mobilenetv2_finetuned_classification_report.txt`
- `metrics/mobilenetv3small_classification_report.txt`
- `metrics/mobilenetv2_finetuned_confusion_matrix.csv`
- `metrics/mobilenetv3small_confusion_matrix.csv`

Графики:

- `plots/mobilenetv2_finetuned_accuracy.png`
- `plots/mobilenetv2_finetuned_loss.png`
- `plots/mobilenetv2_finetuned_confusion_matrix.png`
- `plots/mobilenetv3small_accuracy.png`
- `plots/mobilenetv3small_loss.png`
- `plots/mobilenetv3small_confusion_matrix.png`
- `plots/comparison_accuracy.png`
- `plots/comparison_f1_macro.png`
- `plots/comparison_inference_time.png`
- `plots/comparison_model_size.png`

Экспорт финальной модели:

- `tf_saved_model/final_model/`
- `models/final_model_float32.tflite`
- `models/final_model_quantized.tflite`

В конце `train_compare.py` печатает таблицу:

```text
Model | Test Accuracy | F1 Macro | F1 Weighted | Inference ms/image | FPS | Model Size MB | Params
```

Затем печатается вывод: какая модель точнее, быстрее, легче, какая выбрана
для Raspberry Pi 4 4GB и какая сохранена как `model.h5`.

## Предсказание

```bash
python predict.py --model model.h5 --image path/to/image.jpg
```

Или для конкретного checkpoint:

```bash
python predict.py --model models/mobilenetv3small_finetuned_best.h5 --image path/to/image.jpg
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
