#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laser RX — приёмник лазерного канала: детектор пятна + слотовый ML-декодер
Hamming(7,4) на согласованной фильтрации.

Протокол, разбор алгоритма, результаты, ограничения и настройка параметров —
в README.md. Быстрый запуск:

    python rx.py video2.mp4 --report   # разбор файла, только отчёт
    python rx.py 0                     # живая камера
    python rx.py --help                # все ключи
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

# ==================== ИСТОЧНИК ПО УМОЛЧАНИЮ ====================
DEFAULT_SOURCE = "video2.mp4"

# ==================== ПРОТОКОЛ (секунды) ====================
START_ON = 0.600         # длительность старт-импульса
OFF_DELAY = 0.100        # пауза между старт-битом и первым слотом
BIT_PULSE = 0.200        # длительность импульса единицы
BIT_GAP = 0.050          # пауза после единицы
SLOT = BIT_PULSE + BIT_GAP   # 0.250 — шаг битовой сетки
NBITS = 7
GUARD = 0.250            # после последнего слота ожидаем тишину
PRE_ROLL = 0.150         # перед старт-битом тоже ожидаем тишину

PACKET_SPAN = START_ON + OFF_DELAY + NBITS * SLOT + GUARD   # 2.70 с
# Информативная часть пакета: паузы между битами в счёт не идут (см. _decode).
INFO_DURATION = PRE_ROLL + START_ON + NBITS * BIT_PULSE + GUARD   # 2.40 с
MATURE = PACKET_SPAN + 0.15    # когда пакет можно декодировать
BUFFER_SEC = PACKET_SPAN + 1.2

# Сетка поиска выравнивания. Камера "размазывает" импульс примерно на кадр,
# поэтому фронт детектируется на 30-70 мс раньше реального.
OFFSET_GRID = np.arange(-0.12, 0.1201, 0.010)
SLOT_GRID = np.array([0.240, 0.245, 0.250, 0.255, 0.260])

MIN_IDLE = 0.13          # пауза перед кандидатом на старт (грубый префильтр)
START_ON_FRAC = 0.75     # какая доля старт-окна должна быть засвечена
PRE_IDLE_FRAC = 0.35     # и какая доля тишины перед ним
MEDIAN_TAPS = 3          # медиана по кадрам: убивает одиночные сбои детектора
ARB_WIN = 1.00           # окно, внутри которого кандидаты конкурируют за пакет

# Пороги приёма пакета
MIN_SCORE = 0.905        # доля совпадения с эталоном
MIN_MARGIN = 0.13        # отрыв от второй по счёту цифры
MIN_COVERAGE = 0.85     # какую долю пакета мы обязаны реально увидеть

# ==================== ДЕТЕКТОР ПЯТНА ====================
# Зелёная рамка мишени — по ней вырезаем ROI, чтобы красный корпус робота,
# куртки и прочий фон вообще не участвовали в поиске.
BOARD_HSV_LO = np.array((35, 60, 40), np.uint8)
BOARD_HSV_HI = np.array((90, 255, 255), np.uint8)
BOARD_MIN_AREA = 3000
BOARD_PAD = 0.08         # расширение ROI в долях от размера
# Если мишени в кадре нет, то и пятна на ней быть не может. Держим последний
# ROI лишь несколько кадров (мишень могла смазаться), иначе в устаревшую рамку
# заезжает красная одежда/лица и детектор радостно принимает их за лазер.
ROI_MAX_AGE = 5

# Санитарные ограничения на пятно: лазерная точка мелкая и компактная.
MAX_SPOT_FRAC = 0.04       # не больше 4% площади ROI мишени
MAX_SPOT_FRAC_WIDE = 0.005  # когда ищем по всему кадру — жёстче
MIN_SPOT_FILL = 0.25       # заполнение габаритного прямоугольника
MAX_SPOT_ASPECT = 4.0      # вытянутость

RED_TH_MIN, RED_TH_MAX = 35, 110   # границы адаптивного порога "красности"
RED_TH_MARGIN = 15                 # запас над фоном
BG_ALPHA = 0.05                    # скорость адаптации фона

AREA_LO, AREA_HI = 4.0, 24.0       # мягкий переход "нет пятна" -> "есть пятно"
ON_ENTER, ON_EXIT = 18.0, 6.0      # триггер Шмитта по площади, px
CORE_MIN_V = 235                   # у пятна лазера всегда есть пересвеченное ядро


# ==================== HAMMING(7,4) ====================
def hamming_encode(value: int) -> list[int]:
    """4 бита -> 7 бит в раскладке [p1, p2, d1, p3, d2, d3, d4]."""
    d1, d2, d3, d4 = (value >> 3) & 1, (value >> 2) & 1, (value >> 1) & 1, value & 1
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def hamming_decode(bits) -> tuple[int, int]:
    """7 бит -> (значение, позиция исправленного бита или 0)."""
    b = list(bits)
    p1, p2, d1, p3, d2, d3, d4 = b
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1
    if 1 <= syndrome <= 7:
        b[syndrome - 1] ^= 1
    _, _, d1, _, d2, d3, d4 = b
    return (d1 << 3) | (d2 << 2) | (d3 << 1) | d4, syndrome


# Все 16 корректных кодовых слов — эталоны для согласованной фильтрации.
CODEWORDS = np.array([hamming_encode(v) for v in range(16)], dtype=np.float64)


# ==================== ДЕТЕКТОР ЛАЗЕРНОГО ПЯТНА ====================
class SpotDetector:
    """Ищет пятно лазера и возвращает мягкую (0..1) оценку "лазер горит".

    Пятно лазера на мишени — это яркое белое ядро с красным гало, поэтому
    признак = "краснота" R - max(G, B) плюс наличие пересвеченного ядра.
    Порог красноты адаптируется к фону, так что детектор не надо
    перенастраивать под каждое освещение.
    """

    def __init__(self, mode: str = "auto", fixed_threshold: int | None = None):
        self.mode = mode                  # board | full | auto
        self.fixed_threshold = fixed_threshold
        self.bg_level = None
        self.roi = None
        self.roi_age = 10 ** 9
        self.state = False
        self.threshold = RED_TH_MIN
        self.point = None
        self.area = 0
        self.board_found = False
        self.board_ever = False

    # ---------- поиск мишени ----------
    def _find_board(self, hsv, shape):
        m = cv2.inRange(hsv, BOARD_HSV_LO, BOARD_HSV_HI)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < BOARD_MIN_AREA:
            return None
        x, y, w, h = cv2.boundingRect(c)
        px, py = int(w * BOARD_PAD), int(h * BOARD_PAD)
        H, W = shape[:2]
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(W, x + w + px), min(H, y + h + py)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    # ---------- основной вызов ----------
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        roi = None
        if self.mode in ("board", "auto"):
            roi = self._find_board(hsv, frame.shape)
            self.board_found = roi is not None
            if roi is not None:
                self.roi, self.roi_age = roi, 0
                self.board_ever = True
            else:
                self.roi_age += 1
                if self.roi is not None and self.roi_age <= ROI_MAX_AGE:
                    roi = self.roi          # мишень мигнула — держим прошлый ROI
                else:
                    self.roi = None
        # На весь кадр переходим только если мишени тут вообще не бывает
        # (другая постановка). Если мишень была и пропала — это "не вижу",
        # а не повод искать красное по всему кадру.
        if roi is None and (self.mode == "full" or
                            (self.mode == "auto" and not self.board_ever)):
            H, W = frame.shape[:2]
            roi = (0, 0, W, H)

        # Мишени не видно => неизвестно, горит ли лазер. Именно "неизвестно",
        # а не "не горит": иначе пропавшая мишень выглядит как длинная пауза
        # и складывается в правдоподобный, но выдуманный пакет.
        if roi is None:
            self.state = False
            self.point, self.area = None, 0
            return False, 0.0, False

        x, y, w, h = roi
        sub = frame[y:y + h, x:x + w]
        b, g, r = cv2.split(sub.astype(np.int16))
        red = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
        red = cv2.GaussianBlur(red, (5, 5), 0)

        # Адаптивный порог: фон обновляем только пока лазер считается погашенным,
        # иначе само пятно поднимет порог и детектор "ослепнет".
        if self.fixed_threshold is not None:
            self.threshold = int(self.fixed_threshold)
        else:
            if not self.state:
                level = float(np.percentile(red, 99.0))
                self.bg_level = level if self.bg_level is None else \
                    (1 - BG_ALPHA) * self.bg_level + BG_ALPHA * level
            base = self.bg_level if self.bg_level is not None else RED_TH_MIN
            self.threshold = int(np.clip(base + RED_TH_MARGIN, RED_TH_MIN, RED_TH_MAX))

        mask = (red > self.threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Внутри мишени кроме лазера красного нет, поэтому там можно быть
        # мягче. По всему кадру приходится требовать все признаки точки.
        area, point = self._best_blob(mask, sub, strict=not self.board_found)
        self.area = area
        self.point = (x + point[0], y + point[1]) if point else None

        # Триггер Шмитта — убирает дребезг на фронтах.
        if self.state:
            self.state = area >= ON_EXIT
        else:
            self.state = area >= ON_ENTER

        soft = float(np.clip((area - AREA_LO) / (AREA_HI - AREA_LO), 0.0, 1.0))
        return self.state, soft, True

    def _best_blob(self, mask, sub, strict: bool):
        """Крупнейшее красное пятно, похожее на лазерную точку."""
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return 0, None
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        frac = MAX_SPOT_FRAC_WIDE if strict else MAX_SPOT_FRAC
        max_area = frac * mask.shape[0] * mask.shape[1]
        best_area, best_pt = 0, None
        # Пересвеченное ядро — главный признак лазера: красный корпус робота
        # или куртка красные, но тёмные (серый ~60), а ядро пятна белое.
        need_core = strict
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a <= best_area or a > max_area:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if a < MIN_SPOT_FILL * bw * bh:
                continue
            if max(bw, bh) > MAX_SPOT_ASPECT * max(min(bw, bh), 1):
                continue
            if need_core:
                # Ядро проверяем по габаритному прямоугольнику, а не по самой
                # маске: центр пятна пересвечен в белый, его "краснота" ~0,
                # поэтому в красную маску попадает только гало вокруг него.
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                core = gray[by:by + bh, bx:bx + bw]
                if core.size == 0 or int(core.max()) < CORE_MIN_V:
                    continue
            best_area = a
            best_pt = (int(cents[i][0]), int(cents[i][1]))
        return best_area, best_pt


# ==================== ДЕКОДЕР ПАКЕТА ====================
class PacketDecoder:
    """Копит отсчёты (время, мягкое состояние) и декодирует пакеты целиком.

    Состояние полностью выводится из буфера, поэтому декодер физически
    не может "залипнуть": плохой пакет просто не пройдёт порог, а буфер
    продолжит скользить дальше.
    """

    def __init__(self, min_score=MIN_SCORE, min_margin=MIN_MARGIN, verbose=True):
        self.min_score = min_score
        self.min_margin = min_margin
        self.verbose = verbose
        self.samples: deque = deque()
        self.candidates: deque = deque()
        self._raw: deque = deque(maxlen=MEDIAN_TAPS)
        self.prev_on = False
        self.off_since = None
        self.accepted_until = -1e18
        self.stats = {"candidates": 0, "accepted": 0, "rejected": 0}

    def push(self, t: float, on: bool, soft: float, valid: bool = True):
        """Добавить кадр. Возвращает список принятых пакетов (dict)."""
        # Медиана по 3 кадрам: одиночный ложный кадр (блик, пропуск детекта)
        # больше не рвёт импульс на два и не рождает мусорных фронтов.
        self._raw.append((t, float(soft), 1.0 if on else 0.0, 1.0 if valid else 0.0))
        if len(self._raw) < MEDIAN_TAPS:
            return []
        raw = list(self._raw)
        t = raw[MEDIAN_TAPS // 2][0]
        soft = float(np.median([r[1] for r in raw]))
        on = sum(r[2] for r in raw) * 2 >= MEDIAN_TAPS
        valid = sum(r[3] for r in raw) * 2 >= MEDIAN_TAPS

        self.samples.append((t, soft, on, valid))
        while self.samples and self.samples[0][0] < t - BUFFER_SEC:
            self.samples.popleft()

        if valid and on and not self.prev_on:
            idle = t - self.off_since if self.off_since is not None else 1e9
            if idle >= MIN_IDLE:
                self.candidates.append(t)
        elif not on and self.prev_on:
            self.off_since = t
        self.prev_on = on and valid

        return self._harvest(t)

    def flush(self):
        """Добить кандидатов, оставшихся в буфере (конец файла/остановка)."""
        if not self.samples:
            return []
        return self._harvest(self.samples[-1][0], force=True)

    def _harvest(self, t: float, force: bool = False):
        out = []
        need = MATURE if force else MATURE + ARB_WIN
        # Если рядом несколько кандидатов (например, слабый ложный блик прямо
        # перед настоящим старт-битом), декодируем всех и берём лучшего —
        # иначе мусорный кандидат "съедает" реальный пакет.
        while self.candidates and (t - self.candidates[0]) >= need:
            t0 = self.candidates[0]
            group = []
            while self.candidates and self.candidates[0] <= t0 + ARB_WIN:
                group.append(self.candidates.popleft())

            variants = []
            for tr in group:
                if tr < self.accepted_until:
                    continue                   # внутри уже принятого пакета
                self.stats["candidates"] += 1
                res = self._decode(tr)
                if res is not None:
                    variants.append(res)
            if not variants:
                continue

            res = max(variants, key=lambda r: r["score"])
            if res["score"] >= self.min_score and res["margin"] >= self.min_margin:
                self.accepted_until = res["t"] + PACKET_SPAN - GUARD
                self.stats["accepted"] += 1
                out.append(res)
            else:
                self.stats["rejected"] += 1
                if self.verbose:
                    print(f"[RX] пакет отклонён t={res['t']:.3f} "
                          f"score={res['score']:.3f} margin={res['margin']:.3f} "
                          f"(лучшая гипотеза {res['digit']})")
        return out

    # ---------- согласованная фильтрация ----------
    def _decode(self, tr: float):
        arr = np.fromiter(
            (v for s in self.samples
             for v in (s[0], s[1], 1.0 if s[2] else 0.0, 1.0 if s[3] else 0.0)),
            dtype=np.float64,
        ).reshape(-1, 4)
        t_all, soft_all, on_all, ok_all = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # Окно берём с запасом на сдвиг шаблона.
        pad = float(OFFSET_GRID.max()) + 0.03
        sel = (t_all >= tr - PRE_ROLL - pad) & (t_all < tr + PACKET_SPAN + pad)
        if sel.sum() < 25:
            return None
        ts, soft, ons, oks = t_all[sel], soft_all[sel], on_all[sel], ok_all[sel]

        # Вес отсчёта = длительность кадра: декодер не зависит от fps
        # и переживает выпавшие кадры. Кадры без видимой мишени весят 0 —
        # они не подтверждают ни одну гипотезу.
        dt = np.diff(ts)
        med = float(np.median(dt)) if len(dt) else 0.033
        w = np.append(dt, med)
        w = np.clip(w, 0.0, 4 * med)         # разрыв в записи не должен доминировать
        w = w * oks
        wsum = w.sum()
        if wsum <= 0:
            return None

        def frac(a, b):
            """Доля времени с включённым лазером на интервале [a, b)."""
            ww = w * ((ts >= a) & (ts < b))
            s = ww.sum()
            return float((ons * ww).sum() / s) if s > 0 else 0.0

        # Кандидат должен выглядеть как старт-бит: засвеченное окно 600 мс
        # и тишина перед ним. Проверяем долей, а не непрерывной длиной, —
        # иначе один битый кадр внутри старт-импульса убивает весь пакет.
        if frac(tr, tr + START_ON) < START_ON_FRAC:
            return None
        if frac(tr - PRE_ROLL, tr) > PRE_IDLE_FRAC:
            return None

        # Выравнивание ищем по всей осциллограмме вместе со старт-битом — он
        # даёт надёжный якорь. А вот саму цифру выбираем только по битовой
        # части: старт-бит одинаков у всех 16 гипотез и лишь размывает отрыв.
        best = None
        data_t0_base = START_ON + OFF_DELAY

        for off in OFFSET_GRID:
            rel = ts - (tr + off)
            in_pre = (rel >= -PRE_ROLL) & (rel < 0.0)
            in_start = (rel >= 0.0) & (rel < START_ON)
            for sl in SLOT_GRID:
                k_idx = np.floor((rel - data_t0_base) / sl)
                phase = (rel - data_t0_base) - k_idx * sl
                in_pulse = (rel >= data_t0_base) & (k_idx >= 0) & (k_idx < NBITS) \
                    & (phase < BIT_PULSE)
                kk = np.clip(k_idx, 0, NBITS - 1).astype(np.int64)

                exp = CODEWORDS[:, kk] * in_pulse       # (16, n)
                exp[:, in_start] = 1.0

                # Считаем только информативные участки. Паузы 50 мс после
                # единицы и 100 мс после старт-бита информации не несут, зато
                # камера их регулярно "замазывает" (единицы подряд слипаются
                # в один импульс). Их исключение и убирает эту ошибку.
                data_end = data_t0_base + NBITS * sl
                in_guard = (rel >= data_end) & (rel < data_end + GUARD)
                info = in_pre | in_start | in_pulse | in_guard

                fit = 1.0 - np.abs(soft[None, :] - exp)
                wi = w * info
                wisum = wi.sum()
                wd = w * in_pulse
                wdsum = wd.sum()
                if wisum <= 0 or wdsum <= 0:
                    continue
                # Какая доля пакета вообще наблюдалась. Сравниваем с
                # ожидаемой длительностью, а не с тем, что оказалось в буфере,
                # иначе обрезанный на конце записи пакет выглядит "полным".
                cover = wisum / INFO_DURATION
                if cover < MIN_COVERAGE:
                    continue
                full = fit @ wi / wisum
                data = fit @ wd / wdsum

                if best is None or full.max() > best[0]:
                    best = (float(full.max()), float(off), float(sl), full, data, cover)

        if best is None:
            return None
        _, off, sl, full_vec, data_vec, cover = best
        order = np.argsort(data_vec)[::-1]
        digit = int(order[0])
        margin = float(data_vec[order[0]] - data_vec[order[1]])
        score = float(full_vec[digit])

        bits = self._sample_bits(ts, soft, tr + off, sl)
        raw_digit, syndrome = hamming_decode(bits)
        return {
            "t": tr, "digit": digit, "score": score, "margin": margin,
            "offset": off, "slot": sl, "bits": bits, "coverage": cover,
            "raw_digit": raw_digit, "syndrome": syndrome,
            "runner_up": int(order[1]),
        }

    @staticmethod
    def _sample_bits(ts, soft, t0, sl):
        """Жёсткая нарезка битов по найденной сетке — только для отладки/лога."""
        base = t0 + START_ON + OFF_DELAY
        bits = []
        for k in range(NBITS):
            a = base + k * sl
            m = (ts >= a) & (ts < a + BIT_PULSE)
            bits.append(1 if (m.any() and soft[m].mean() >= 0.5) else 0)
        return bits


# ==================== ПРИЁМНИК ====================
class Receiver:
    def __init__(self, source, mode="auto", display=True, realtime=False,
                 fixed_threshold=None, loop=False, verbose=True):
        self.source_arg = source
        self.is_file = not str(source).isdigit()
        self.source = source if self.is_file else int(source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps_nominal = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.detector = SpotDetector(mode=mode, fixed_threshold=fixed_threshold)
        self.decoder = PacketDecoder(verbose=verbose)
        self.display = display
        self.realtime = realtime
        self.loop = loop
        self.verbose = verbose

        self.frame_idx = 0
        self.fps = 0.0
        self._fps_t0 = time.time()
        self.history: deque = deque(maxlen=200)
        self.results: list = []
        self.last_digit = None
        self.repeat = 0

    # ---------- время ----------
    def timestamp(self) -> float:
        """Файл -> время видео, камера -> системное время.

        Это принципиально: при разборе файла обработка идёт не в реальном
        времени, и time.time() даёт полную кашу в длительностях импульсов.
        """
        if self.is_file:
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms and ms > 0:
                return ms / 1000.0
            return self.frame_idx / max(self.fps_nominal, 1.0)
        return time.time()

    def run(self):
        src = self.source_arg
        print(f"[RX] источник: {src}  режим ROI: {self.detector.mode}")
        if self.detector.mode == "full":
            print("[RX] ВНИМАНИЕ: режим full ищет пятно по всему кадру и заметно "
                  "менее надёжен.\n"
                  "     Если в кадре есть зелёная мишень — используйте --roi auto.")
        print(f"[RX] ожидаемый пакет: старт {START_ON*1000:.0f} мс + "
              f"{NBITS} слотов по {SLOT*1000:.0f} мс")
        if self.display:
            print("[RX] 'q' — выход, 'r' — сброс декодера")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    continue
                if self.is_file:
                    print("[RX] конец файла")
                    break
                # Живая камера: разрыв не должен убивать процесс.
                print("[RX] кадр не получен, переподключение…")
                time.sleep(0.2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    time.sleep(1.0)
                continue

            t = self.timestamp()
            self.frame_idx += 1

            try:
                on, soft, valid = self.detector.detect(frame)
                packets = self.decoder.push(t, on, soft, valid)
            except Exception as e:                      # noqa: BLE001
                print(f"[RX] ошибка обработки кадра {self.frame_idx}: {e}")
                continue

            self.history.append((t, soft, on, valid))
            for p in packets:
                self._on_packet(p)

            if self.frame_idx % 30 == 0:
                now = time.time()
                self.fps = 30.0 / max(now - self._fps_t0, 1e-6)
                self._fps_t0 = now

            if self.display:
                self._draw(frame, on)
                delay = int(1000 / max(self.fps_nominal, 1)) if self.realtime else 1
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('r'):
                    self.decoder = PacketDecoder(verbose=self.verbose)
                    print("[RX] декодер сброшен вручную")

        for p in self.decoder.flush():      # не теряем пакет в самом конце записи
            self._on_packet(p)
        self.cap.release()
        if self.display:
            cv2.destroyAllWindows()
        self._summary()

    def _on_packet(self, p):
        self.results.append(p)
        if p["digit"] == self.last_digit:
            self.repeat += 1
        else:
            self.last_digit, self.repeat = p["digit"], 1
        bits = "".join(map(str, p["bits"]))
        print(f"\n>>> ЦИФРА {p['digit']}  (t={p['t']:.2f}s, score={p['score']:.3f}, "
              f"margin={p['margin']:.3f}, подряд×{self.repeat})")
        print(f"    сетка: сдвиг {p['offset']*1000:+.0f} мс, слот {p['slot']*1000:.0f} мс, "
              f"биты {bits}" + (f", Hamming исправил позицию {p['syndrome']}"
                                if p["syndrome"] else ""))

    # ---------- визуализация ----------
    def _draw(self, frame, on):
        vis = frame
        roi = self.detector.roi if self.detector.mode != "full" else None
        if roi is not None:
            x, y, w, h = roi
            colour = (0, 200, 0) if self.detector.board_found else (0, 140, 200)
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
        if self.detector.point is not None and on:
            cv2.circle(vis, self.detector.point, 18, (0, 255, 255), 2)
            cv2.drawMarker(vis, self.detector.point, (0, 255, 255),
                           cv2.MARKER_CROSS, 12, 1)

        H, W = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (W, 58), (0, 0, 0), -1)
        st = "LASER ON" if on else "laser off"
        cv2.putText(vis, st, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if on else (120, 120, 120), 2)
        cv2.putText(vis, f"area={self.detector.area:5d}  th={self.detector.threshold:3d}"
                         f"  fps={self.fps:4.1f}", (170, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        s = self.decoder.stats
        cv2.putText(vis, f"packets ok={s['accepted']} rejected={s['rejected']}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if self.last_digit is not None:
            cv2.putText(vis, f"DIGIT {self.last_digit} x{self.repeat}", (W - 250, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        self._draw_wave(vis)
        cv2.imshow("Laser RX", vis)

    def _draw_wave(self, vis):
        """Осциллограмма последних кадров — сразу видно, что ловит детектор."""
        H, W = vis.shape[:2]
        h, pad = 70, 10
        y1 = H - pad
        y0 = y1 - h
        cv2.rectangle(vis, (pad, y0), (W - pad, y1), (0, 0, 0), -1)
        pts = list(self.history)
        if len(pts) < 2:
            return
        n = len(pts)
        for i, (_, soft, st, ok) in enumerate(pts):
            x = pad + int((W - 2 * pad - 1) * i / (n - 1))
            if not ok:
                cv2.line(vis, (x, y1), (x, y0 + 2), (60, 60, 60), 1)
                continue
            yy = y1 - int((h - 4) * soft)
            cv2.line(vis, (x, y1), (x, yy), (0, 255, 0) if st else (0, 90, 160), 1)
        cv2.rectangle(vis, (pad, y0), (W - pad, y1), (90, 90, 90), 1)

    def _summary(self):
        s = self.decoder.stats
        print("\n================ ИТОГ ================")
        print(f"кадров обработано : {self.frame_idx}")
        print(f"кандидатов старта : {s['candidates']}")
        print(f"принято пакетов   : {s['accepted']}")
        print(f"отклонено пакетов : {s['rejected']}")
        if self.results:
            digits = [r["digit"] for r in self.results]
            print(f"цифры             : {digits}")
            vals, cnts = np.unique(digits, return_counts=True)
            print("распределение     : " +
                  ", ".join(f"{v}×{c}" for v, c in zip(vals, cnts)))
            sc = np.array([r["score"] for r in self.results])
            mg = np.array([r["margin"] for r in self.results])
            print(f"score  min/сред   : {sc.min():.3f} / {sc.mean():.3f}")
            print(f"margin min/сред   : {mg.min():.3f} / {mg.mean():.3f}")
        print("======================================")


def main():
    ap = argparse.ArgumentParser(description="Приёмник лазерного канала Hamming(7,4)")
    ap.add_argument("source", nargs="?", default=DEFAULT_SOURCE,
                    help="путь к видео или индекс камеры (по умолчанию %(default)s)")
    ap.add_argument("--roi", choices=("board", "full", "auto"), default="auto",
                    help="board — только внутри зелёной мишени, full — весь кадр, "
                         "auto — мишень с откатом на весь кадр")
    ap.add_argument("--no-display", action="store_true", help="без окон OpenCV")
    ap.add_argument("--report", action="store_true",
                    help="быстрый прогон без окон, только итоговый отчёт")
    ap.add_argument("--realtime", action="store_true",
                    help="проигрывать файл со скоростью съёмки")
    ap.add_argument("--loop", action="store_true", help="повторять файл по кругу")
    ap.add_argument("--red-threshold", type=int, default=None,
                    help="фиксированный порог красноты вместо адаптивного")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    a = ap.parse_args()

    try:
        rx = Receiver(
            a.source, mode=a.roi,
            display=not (a.no_display or a.report),
            realtime=a.realtime, fixed_threshold=a.red_threshold,
            loop=a.loop, verbose=not a.report,
        )
        rx.decoder.min_score = a.min_score
        rx.decoder.min_margin = a.min_margin
        rx.run()
    except KeyboardInterrupt:
        print("\n[RX] остановлено пользователем")
    except Exception as e:                      # noqa: BLE001
        print(f"[RX] ФАТАЛЬНАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
