"""Generate a short Russian PDF report for the competition package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.pdf for the competition package.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("neural_network_project/metrics/mobilenetv2_finetuned_metrics.json"),
        help="Path to MobileNetV2 fine-tuned metrics JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.pdf"),
        help="Output PDF path.",
    )
    return parser.parse_args()


def import_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
            Paragraph,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: reportlab. Install it with: "
            "python -m pip install -r neural_network_project/requirements.txt"
        ) from exc

    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "cm": cm,
        "pdfmetrics": pdfmetrics,
        "TTFont": TTFont,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
        "Paragraph": Paragraph,
    }


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {path}. Run train_compare.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def register_cyrillic_font(deps: dict) -> tuple[str, str]:
    pdfmetrics = deps["pdfmetrics"]
    ttfont = deps["TTFont"]

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/tahomabd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]

    regular = next((path for path in candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), regular)
    if regular is None:
        raise FileNotFoundError("No Cyrillic TrueType font found for PDF generation.")

    pdfmetrics.registerFont(ttfont("ReportFont", str(regular)))
    pdfmetrics.registerFont(ttfont("ReportFont-Bold", str(bold)))
    return "ReportFont", "ReportFont-Bold"


def format_float(value, digits: int = 4) -> str:
    if value is None:
        return "нет данных"
    return f"{float(value):.{digits}f}"


def format_int(value) -> str:
    if value is None:
        return "нет данных"
    return f"{int(value):,}".replace(",", " ")


def build_report(output_path: Path, metrics: dict) -> None:
    deps = import_reportlab()
    font_name, bold_font_name = register_cyrillic_font(deps)

    colors = deps["colors"]
    styles = deps["getSampleStyleSheet"]()
    paragraph_style = deps["ParagraphStyle"]
    paragraph = deps["Paragraph"]
    spacer = deps["Spacer"]
    table = deps["Table"]
    table_style = deps["TableStyle"]
    document = deps["SimpleDocTemplate"](
        str(output_path),
        pagesize=deps["A4"],
        rightMargin=1.5 * deps["cm"],
        leftMargin=1.5 * deps["cm"],
        topMargin=1.4 * deps["cm"],
        bottomMargin=1.4 * deps["cm"],
        title="Отчёт по модели классификации чёрного ящика",
    )

    title_style = paragraph_style(
        "TitleRu",
        parent=styles["Title"],
        fontName=bold_font_name,
        fontSize=15,
        leading=18,
        alignment=deps["TA_CENTER"],
        spaceAfter=10,
    )
    heading_style = paragraph_style(
        "HeadingRu",
        parent=styles["Heading2"],
        fontName=bold_font_name,
        fontSize=11,
        leading=13,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = paragraph_style(
        "BodyRu",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=12,
        spaceAfter=5,
    )
    small_style = paragraph_style(
        "SmallRu",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8,
        leading=10,
        spaceAfter=3,
    )

    story = [
        paragraph("Отчёт по модели классификации изображений", title_style),
        paragraph("Команда: А2026-ИИ-Ёжики_на_паскале", body_style),
        paragraph("1. Цель работы", heading_style),
        paragraph(
            "Цель работы - разработать модель компьютерного зрения для определения "
            "наличия макета чёрного ящика и распознавания цифры на нём. Модель решает "
            "задачу классификации на 10 классов: девять классов цифр и отдельный класс "
            "отсутствия объекта.",
            body_style,
        ),
        paragraph("2. Описание датасета", heading_style),
        paragraph(
            "Для обучения используется рабочий датасет с 10 классами: "
            "digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7, "
            "digit_8, digit_9 и no_object. Для сдачи подготовлен архив dataset.zip, "
            "в котором структура приведена к требованиям регламента: dataset/black_box "
            "и dataset/no_object. При этом рабочая структура с отдельными классами цифр "
            "сохраняется для обучения.",
            body_style,
        ),
        paragraph(
            "Исходные изображения не уменьшаются физически. Проверяемое минимальное "
            "разрешение исходных изображений - не меньше 640x480. Масштабирование до "
            "224x224 выполняется только внутри preprocessing pipeline при подаче в модель.",
            body_style,
        ),
        paragraph("3. Архитектура модели", heading_style),
        paragraph(
            "Финальная модель: MobileNetV2 fine-tuned. Используется transfer learning "
            "с весами ImageNet. Вход модели имеет размер 224x224x3. Последний слой "
            "заменён на классификатор с softmax для 10 классов.",
            body_style,
        ),
        paragraph("4. Обучение", heading_style),
        paragraph(
            "Датасет разделён на train/val/test. На этапе обучения применяются безопасные "
            "аугментации: небольшой rotation, zoom, brightness и contrast. Horizontal flip "
            "не используется, так как зеркальное отражение может изменить визуальный образ "
            "цифр. После начального обучения выполнен fine-tuning последних слоёв backbone "
            "с малым learning rate.",
            body_style,
        ),
        paragraph("5. Метрики финальной модели", heading_style),
    ]

    metrics_rows = [
        ["Метрика", "Значение"],
        ["Test accuracy", format_float(metrics.get("test_accuracy"))],
        ["F1 macro", format_float(metrics.get("f1_macro"))],
        ["F1 weighted", format_float(metrics.get("f1_weighted"))],
        ["Inference ms/image", format_float(metrics.get("inference_ms_per_image"), 2)],
        ["FPS", format_float(metrics.get("fps"), 2)],
        ["Model size MB", format_float(metrics.get("model_size_mb"), 2)],
        ["Params", format_int(metrics.get("params"))],
    ]
    metrics_table = table(metrics_rows, colWidths=[7.0 * deps["cm"], 8.0 * deps["cm"]])
    metrics_table.setStyle(
        table_style(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metrics_table, spacer(1, 0.2 * deps["cm"])])

    story.extend(
        [
            paragraph("6. Развёртывание на Raspberry Pi 4 4GB", heading_style),
            paragraph(
                "Для Raspberry Pi 4 4GB рекомендуется использовать экспортированную "
                "TensorFlow Lite модель. Float32-вариант подходит для проверки качества, "
                "а quantized TFLite с dynamic range quantization предпочтителен для "
                "ускорения инференса и уменьшения размера файла. Перед запуском на "
                "устройстве необходимо сохранить тот же preprocessing: RGB, resize до "
                "224x224, подача изображения в модель в ожидаемом диапазоне значений.",
                body_style,
            ),
            paragraph("7. Список файлов для проверки", heading_style),
            paragraph(
                "В комплекте для проверки должны присутствовать: model.h5, model.nnp, "
                "config.json, predict.py и dataset.zip. Дополнительно доступны метрики, "
                "графики, TensorFlow SavedModel и TFLite-экспорты в папке "
                "neural_network_project.",
                body_style,
            ),
            paragraph("Проверяемые файлы:", small_style),
        ]
    )

    files_rows = [
        ["Файл", "Назначение"],
        ["model.h5", "финальная модель Keras"],
        ["model.nnp", "описание архитектуры модели"],
        ["config.json", "конфигурация и итоговые метрики"],
        ["predict.py", "скрипт инференса по одному изображению"],
        ["dataset.zip", "датасет в структуре black_box/no_object"],
    ]
    files_table = table(files_rows, colWidths=[5.0 * deps["cm"], 10.0 * deps["cm"]])
    files_table.setStyle(
        table_style(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(files_table)

    document.build(story)


def main() -> int:
    args = parse_args()
    root = project_root()
    metrics_path = resolve_path(root, args.metrics)
    output_path = resolve_path(root, args.output)
    metrics = load_metrics(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_report(output_path, metrics)
    print(f"Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
