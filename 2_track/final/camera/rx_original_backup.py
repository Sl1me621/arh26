# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Laser Receiver - Time-based Hamming(7,4) decoder
# Читает камеру, детектит лазер, декодирует по временным интервалам
# """

# import cv2
# import numpy as np
# import time
# import sys
# from collections import deque


# # ==================== CONFIG ====================
# CAMERA_INDEX = "C:/Users/Aleksandr/Pictures/video.mp4"                    # /dev/video0
# LASER_HSV_LOWER = np.array((255, 51, 80.0), dtype=np.uint8)
# LASER_HSV_UPPER = np.array((60, 255, 255), dtype=np.uint8)
# MIN_CONTOUR_AREA = 5                # Minimum laser spot area (px)
# MAX_CONTOUR_AREA = 300              # Maximum laser spot area (px)

# # Timing constants (seconds)
# START_MIN, START_MAX = 0.50, 0.70    # Start bit: 600ms ±70ms
# BIT_MIN, BIT_MAX = 0.13, 0.27        # Data bit: 200ms ±70ms
# RECEIVE_TIMEOUT = 2.0                # Reset if no edge for 2s

# # ==================== HAMMING DECODER ====================
# def decode_hamming_7(bits):
#     if len(bits) != 7:
#         return None
#     p1, p2, d1, p3, d2, d3, d4 = bits
#     s1 = p1 ^ d1 ^ d2 ^ d4
#     s2 = p2 ^ d1 ^ d3 ^ d4
#     s3 = p3 ^ d2 ^ d3 ^ d4
    
#     error_bit = (s3 << 2) | (s2 << 1) | s1
#     if error_bit != 0:
#         bit_to_flip = error_bit - 1
#         if bit_to_flip < 7:
#             bits[bit_to_flip] ^= 1
#             print(f"[HAMMING] Исправлена ошибка в бите на позиции {error_bit}")
            
#     _, _, d1, _, d2, d3, d4 = bits
#     return (d1 << 3) | (d2 << 2) | (d3 << 1) | d4

# # ==================== ВРЕМЕННОЙ ДЕКОДЕР ====================
# class LaserTimeDecoder:
#     def __init__(self):
#         self.receiving = False
#         self.waiting_for_start = True
#         self.last_edge_time = None
#         self.last_state = False
#         self.bit_buffer = []

#     def process_frame(self, laser_on: bool, timestamp: float):
#         if laser_on != self.last_state:
#             now = timestamp
#             duration = now - self.last_edge_time if self.last_edge_time else 0
#             self.last_edge_time = now
#             self.last_state = laser_on

#             if self.waiting_for_start:
#                 if laser_on == False: 
#                     if START_MIN <= duration <= START_MAX:
#                         print(f"[DECODER] Пойман СТАРТ-бит: {duration * 1000:.0f} мс")
#                         self.receiving = True
#                         self.waiting_for_start = False
#                         self.bit_buffer = []
#                     else:
#                         return None
                        
#             elif self.receiving:
#                 if BIT_MIN <= duration <= BIT_MAX:
#                     bit = 1 if not laser_on else 0 
#                     self.bit_buffer.append(bit)
#                     print(f"[DECODER] Бит {len(self.bit_buffer)}/7: {bit} ({duration * 1000:.0f} мс)")
                    
#                     if len(self.bit_buffer) == 7:
#                         digit = decode_hamming_7(self.bit_buffer)
#                         print(f"[DECODER] *** УСПЕШНО ДЕКОДИРОВАНА ЦИФРА: {digit} ***")
#                         self._reset()
#                         return digit
#                 else:
#                     print(f"[DECODER] Сбой тайминга: {duration * 1000:.0f} мс. Сброс пакета.")
#                     self._reset()
#                     return None

#         if self.receiving and self.last_edge_time:
#             if (timestamp - self.last_edge_time) > RECEIVE_TIMEOUT:
#                 print("[DECODER] Таймаут ожидания. Сброс.")
#                 self._reset()
#                 return None

#     def _reset(self):
#         self.receiving = False
#         self.waiting_for_start = True
#         self.bit_buffer = []
#         self.last_edge_time = None
#         self.last_state = False

# # ==================== ОСНОВНОЙ КЛАСС КАМЕРЫ ====================
# class LaserDetector:
#     def __init__(self):
#         self.cap = cv2.VideoCapture(CAMERA_INDEX)
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Не удалось открыть камеру {CAMERA_INDEX}")
            
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)
        
#         self.decoder = LaserTimeDecoder()
#         self.frame_count = 0
#         self.fps_start = time.time()
#         self.fps = 0

#     def detect_laser(self, frame):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, LASER_HSV_LOWER, LASER_HSV_UPPER)

#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

#         # ДОПОЛНИТЕЛЬНЫЕ ОКНА ДЛЯ ОТЛАДКИ ФИЛЬТРОВ
#         cv2.imshow('HSV Image', hsv)
#         cv2.imshow('Black/White Mask', mask)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         laser_on = False
#         best_contour = None
#         max_area = 0

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
#                 if area > max_area:
#                     max_area = area
#                     best_contour = cnt

#         annotated = frame.copy()
#         if best_contour is not None:
#             laser_on = True
#             (x, y), radius = cv2.minEnclosingCircle(best_contour)
#             center = (int(x), int(y))
#             radius = int(radius)
#             cv2.circle(annotated, center, radius, (0, 255, 0), 2)
#             cv2.putText(annotated, "LASER", (center-30, center[5]-radius-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
#         return laser_on, annotated

#     def run(self):
#         print("[ПРИЕМНИК] Запуск детектора... (Нажмите 'q' для выхода)")
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("[ПРИЕМНИК] Ошибка чтения кадра")
#                 break
                
#             timestamp = time.time()
#             self.frame_count += 1
            
#             if self.frame_count % 30 == 0:
#                 elapsed = timestamp - self.fps_start
#                 self.fps = 30 / elapsed if elapsed > 0 else 0
#                 self.fps_start = timestamp

#             laser_on, annotated = self.detect_laser(frame)
#             digit = self.decoder.process_frame(laser_on, timestamp)

#             if digit is not None:
#                 cv2.putText(annotated, f"ZONE DECODED: {digit}", (10, 80), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#                 print(f"\n>>> ПРИНЯТА КОМАНДА ЗОНЫ: {digit} <<<\n")

#             status = "RECEIVING" if self.decoder.receiving else ("WAITING START" if self.decoder.waiting_for_start else "IDLE")
#             cv2.putText(annotated, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, annotated.shape-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
#             cv2.putText(annotated, f"Laser: {'ON' if laser_on else 'OFF'}", (10, annotated.shape-40), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if laser_on else (0, 0, 255), 2)

#             cv2.imshow('Laser Receiver', annotated)
            
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('r'):
#                 self.decoder._reset()
#                 print("[ПРИЕМНИК] Ручной сброс состояния")

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     detector = LaserDetector()
#     detector.run()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import sys
from collections import deque

# ==================== CONFIG ====================
CAMERA_INDEX = "C:/Users/Aleksandr/Pictures/video2.mp4"

# --- ВАРИАНТ 1: ОРИГИНАЛЬНАЛЬНЫЙ ФИЛЬТР (Поиск яркого/засвеченного пятна) ---
LASER_HSV_LOWER = np.array((255, 51, 80.0), dtype=np.uint8)
LASER_HSV_UPPER = np.array((60, 255, 255), dtype=np.uint8)

# --- ВАРИАНТ 2: ФИЛЬТР НА КРАСНЫЙ ЦВЕТ (Раскомментируйте, если затемнили камеру) ---
# LASER_HSV_LOWER = np.array(, dtype=np.uint8)
# LASER_HSV_UPPER = np.array(, dtype=np.uint8)

MIN_CONTOUR_AREA = 5 
MAX_CONTOUR_AREA = 500 

# Timing constants (seconds)
START_MIN, START_MAX = 0.50, 0.70 
BIT_MIN, BIT_MAX = 0.13, 0.27 
RECEIVE_TIMEOUT = 2.0 

# ==================== HAMMING DECODER ====================
def decode_hamming_7(bits):
    """Classic Hamming(7,4) decoder - corrects 1-bit errors"""
    if len(bits) != 7:
        return None
    p1, p2, d1, p3, d2, d3, d4 = bits
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    
    error_bit = (s3 << 2) | (s2 << 1) | s1
    if error_bit != 0:
        bit_to_flip = error_bit - 1
        if bit_to_flip < 7:
            bits[bit_to_flip] ^= 1
            print(f"[HAMMING] Corrected 1-bit error at position {error_bit}")
            
    _, _, d1, _, d2, d3, d4 = bits
    return (d1 << 3) | (d2 << 2) | (d3 << 1) | d4

# ==================== TIME-BASED DECODER ====================
class LaserTimeDecoder:
    def __init__(self):
        self.receiving = False
        self.waiting_for_start = True
        self.last_edge_time = None
        self.last_state = False
        self.bit_buffer = []
        self.start_time = None

    def process_frame(self, laser_on: bool, timestamp: float):
        if laser_on != self.last_state:
            now = timestamp
            duration = now - self.last_edge_time if self.last_edge_time else 0
            self.last_edge_time = now
            self.last_state = laser_on

            if self.waiting_for_start:
                if laser_on == False: 
                    if START_MIN <= duration <= START_MAX:
                        print(f"[DECODER] START bit detected: {duration * 1000:.0f}ms")
                        self.receiving = True
                        self.waiting_for_start = False
                        self.bit_buffer = []
                        self.start_time = now
                    else:
                        print(f"[DECODER] Invalid start pulse: {duration* 1000:.0f}ms (expected 500-700ms)")
                        return None
                        
            elif self.receiving:
                if BIT_MIN <= duration <= BIT_MAX:
                    bit = 1 if not laser_on else 0 
                    self.bit_buffer.append(bit)
                    print(f"[DECODER] Bit {len(self.bit_buffer)}/7: {bit} (duration: {duration*1000:.0f}ms)")
                    
                    if len(self.bit_buffer) == 7:
                        digit = decode_hamming_7(self.bit_buffer)
                        print(f"[DECODER] *** DECODED DIGIT: {digit} ***")
                        self._reset()
                        return digit
                else:
                    print(f"[DECODER] Timing error: {duration * 1000:.0f}ms (expected 130-270ms). Resetting.")
                    self._reset()
                    return None

        if self.receiving and self.last_edge_time:
            if (timestamp - self.last_edge_time) > RECEIVE_TIMEOUT:
                print(f"[DECODER] Timeout ({RECEIVE_TIMEOUT}s). Resetting.")
                self._reset()
                return None

    def _reset(self):
        self.receiving = False
        self.waiting_for_start = True
        self.bit_buffer = []
        self.last_edge_time = None
        self.last_state = False

# ==================== LASER DETECTOR ====================
class LaserDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.decoder = LaserTimeDecoder()
        self.frame_count = 0
        self.fps_start = time.time()
        self.fps = 0

    def detect_laser(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LASER_HSV_LOWER, LASER_HSV_UPPER)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # ДОПОЛНИТЕЛЬНЫЕ ОКНА (HSV И ЧЕРНО-БЕЛАЯ МАСКА ДЛЯ ОТЛАДКИ)
        cv2.imshow('HSV Image', hsv)
        cv2.imshow('Black/White Mask', mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laser_on = False
        best_contour = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
                if area > max_area:
                    max_area = area
                    best_contour = cnt

        annotated = frame.copy()
        if best_contour is not None:
            laser_on = True
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(annotated, center, radius, (0, 255, 0), 2)
            cv2.putText(annotated, "LASER", (center-30, center-radius-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        return laser_on, annotated

    def run(self):
        print("[RECEIVER] Starting laser detection loop...")
        print("[RECEIVER] Press 'q' to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[RECEIVER] Failed to read frame")
                break
                
            timestamp = time.time()
            self.frame_count += 1
            
            if self.frame_count % 30 == 0:
                elapsed = timestamp - self.fps_start
                self.fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start = timestamp

            laser_on, annotated = self.detect_laser(frame)
            digit = self.decoder.process_frame(laser_on, timestamp)

            if digit is not None:
                cv2.putText(annotated, f"DECODED: {digit}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                print(f"\n>>> RECEIVED DIGIT: {digit} <<<\n")

            status = "RECEIVING" if self.decoder.receiving else ("WAITING START" if self.decoder.waiting_for_start else "IDLE")
            cv2.putText(annotated, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ИСПРАВЛЕННЫЙ ВЫВОД ТЕКСТА (с использованием индекса  для кортежа shape)
            h, w, _ = annotated.shape
            cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Laser: {'ON' if laser_on else 'OFF'}", (10, h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if laser_on else (0, 0, 255), 2)

            cv2.imshow('Laser Receiver', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.decoder._reset()
                print("[RECEIVER] Manual reset")

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        detector = LaserDetector()
        detector.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
