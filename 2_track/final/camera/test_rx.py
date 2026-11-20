#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Синтетический тест декодера: в video2.mp4 передаётся только цифра 5, поэтому
остальные 15 значений проверяем на смоделированном сигнале.

Модель камеры и таблица сценариев — в README.md. Запуск: python test_rx.py
"""

import random
import sys

import rx


def make_intervals(digit, t0=1.0):
    """Интервалы включения лазера для одной посылки — как в laser_tx.py."""
    bits = rx.hamming_encode(digit)
    ivs = [(t0, t0 + rx.START_ON)]
    base = t0 + rx.START_ON + rx.OFF_DELAY
    for k, b in enumerate(bits):
        if b:
            s = base + k * rx.SLOT
            ivs.append((s, s + rx.BIT_PULSE))
    return ivs


def simulate(ivs, rng, fps=30.0, drop=0.0, noise=0.0, jitter=0.0, duration=6.0):
    """Кадры камеры: (t, on). drop — доля потерянных кадров, noise — сбои детектора."""
    dt = 1.0 / fps
    out = []
    t = 0.0
    while t < duration:
        a, b = t, t + dt
        on = any(s < b and e > a for s, e in ivs)
        if rng.random() >= drop:
            if rng.random() < noise:
                on = not on
            out.append((t + rng.uniform(-jitter, jitter), on))
        t += dt
    return out


def run_case(digit, rng, **kw):
    ivs = make_intervals(digit)
    frames = simulate(ivs, rng, **kw)
    dec = rx.PacketDecoder(verbose=False)
    got = []
    for t, on in frames:
        for p in dec.push(t, on, 1.0 if on else 0.0):
            got.append(p["digit"])
    return got


def scenario(name, trials=8, **kw):
    rng = random.Random(1234)
    ok = wrong = missed = extra = 0
    bad = []
    for digit in range(16):
        for _ in range(trials):
            got = run_case(digit, rng, **kw)
            if not got:
                missed += 1
                bad.append((digit, "нет пакета"))
            elif got[0] == digit:
                ok += 1
                extra += len(got) - 1
            else:
                wrong += 1
                bad.append((digit, f"получено {got}"))
    total = 16 * trials
    print(f"{name:38s} верно {ok:3d}/{total}  ошибок {wrong}  пропущено {missed}"
          + (f"  лишних {extra}" if extra else ""))
    for d, why in bad[:6]:
        print(f"      цифра {d}: {why}")
    return wrong, missed


def main():
    print(f"кодовые слова: " + ", ".join(
        f"{v}={''.join(map(str, rx.hamming_encode(v)))}" for v in range(16)))
    print()

    fails = 0
    for name, kw in (
        ("30 fps, идеально", dict(jitter=0.0)),
        ("30 fps, дрожание кадров 8 мс", dict(jitter=0.008)),
        ("30 fps, 10% кадров потеряно", dict(drop=0.10, jitter=0.008)),
        ("30 fps, 3% сбоев детектора", dict(noise=0.03, jitter=0.008)),
        ("15 fps (слабая камера)", dict(fps=15.0, jitter=0.01)),
        ("60 fps", dict(fps=60.0, jitter=0.005)),
    ):
        w, _ = scenario(name, **kw)
        fails += w

    # Заведомо за пределами протокола (он рассчитан на 30 fps): полезно знать,
    # как деградирует, но в зачёт не идёт.
    print()
    print("--- вне расчётных условий, справочно ---")
    scenario("15 fps + 10% drop + 3% шум",
             fps=15.0, drop=0.10, noise=0.03, jitter=0.01)

    print()
    if fails == 0:
        print("OK: в расчётных условиях ни одной неверно декодированной цифры")
        return 0
    print(f"ВНИМАНИЕ: неверных декодирований — {fails}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
