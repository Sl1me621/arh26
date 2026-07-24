import cv2
import numpy as np
import random
import argparse
from pathlib import Path

IMG_W = 640
IMG_H = 480
MAX_RETRIES = 100


def parse_digits(value: str):
    value = value.strip()
    if value == "0-9":
        return [str(i) for i in range(10)]
    if value == "1-9":
        return [str(i) for i in range(1, 10)]

    return [ch for ch in value if ch.isdigit()]


def order_quad_tl_tr_br_bl(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def polygon_area(pts):
    pts = np.array(pts, dtype=np.float32)
    return abs(cv2.contourArea(pts))


def quad_aspect_ratio(quad):
    quad = np.array(quad, dtype=np.float32)

    w1 = np.linalg.norm(quad[1] - quad[0])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[3] - quad[0])
    h2 = np.linalg.norm(quad[2] - quad[1])

    w = (w1 + w2) / 2
    h = (h1 + h2) / 2

    if min(w, h) < 1:
        return 999

    return max(w / h, h / w)


def is_convex_quad(quad):
    quad = np.array(quad, dtype=np.float32)
    return cv2.isContourConvex(quad.astype(np.int32))


def is_good_cube(proj_pts, visible):
    """
    Проверка, чтобы не появлялись сломанные кубы,
    длинные треугольники, сильные растяжения и вылеты за кадр.
    """
    proj_pts = np.array(proj_pts, dtype=np.float32)

    # 1. Все вершины должны быть внутри кадра с небольшим запасом
    margin = 20
    if np.any(proj_pts[:, 0] < margin):
        return False
    if np.any(proj_pts[:, 0] > IMG_W - margin):
        return False
    if np.any(proj_pts[:, 1] < margin):
        return False
    if np.any(proj_pts[:, 1] > IMG_H - margin):
        return False

    # 2. Размер куба в кадре должен быть адекватным
    min_xy = proj_pts.min(axis=0)
    max_xy = proj_pts.max(axis=0)
    bbox_w = max_xy[0] - min_xy[0]
    bbox_h = max_xy[1] - min_xy[1]

    if bbox_w < 120 or bbox_h < 100:
        return False

    if bbox_w > 430 or bbox_h > 380:
        return False

    # 3. Должно быть видно хотя бы 2 грани
    if len(visible) < 2:
        return False

    # 4. Нижнюю грань лучше не показывать
    visible_names = [face_name for face_name, _, _ in visible]
    if "bottom" in visible_names:
        return False

    # 5. Проверяем каждую видимую грань
    total_area = 0

    for face_name, idxs, _ in visible:
        quad = order_quad_tl_tr_br_bl(proj_pts[idxs])

        area = polygon_area(quad)
        aspect = quad_aspect_ratio(quad)

        if not is_convex_quad(quad):
            return False

        # слишком маленькая грань
        if area < 2500:
            return False

        # слишком огромная грань
        if area > 85000:
            return False

        # слишком вытянутая грань
        if aspect > 3.8:
            return False

        total_area += area

    # 6. Общая площадь видимых граней тоже должна быть нормальной
    if total_area < 12000:
        return False

    if total_area > 150000:
        return False

    return True


def rotation_matrix_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=np.float32)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=np.float32)

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


def project_points(points_3d, K):
    pts = points_3d.copy()
    z = pts[:, 2:3]

    pts_2d = pts @ K[:3, :3].T
    pts_2d[:, 0] /= z[:, 0]
    pts_2d[:, 1] /= z[:, 0]

    return pts_2d[:, :2]


def create_background(underwater=False):
    if not underwater:
        val = random.randint(235, 250)
        return np.full((IMG_H, IMG_W, 3), (val, val, val), dtype=np.uint8)

    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    top_color = np.array([
        random.randint(150, 200),
        random.randint(170, 210),
        random.randint(170, 220)
    ], dtype=np.float32)

    bottom_color = np.array([
        random.randint(70, 110),
        random.randint(100, 140),
        random.randint(110, 160)
    ], dtype=np.float32)

    for y in range(IMG_H):
        t = y / (IMG_H - 1)
        color = top_color * (1 - t) + bottom_color * t
        img[y, :] = color

    return img


def add_underwater_effect(img):
    out = img.copy()

    if random.random() < 0.9:
        k = random.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (k, k), 0)

    noise = np.random.normal(0, random.uniform(3, 10), out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    overlay = out.copy()
    for _ in range(random.randint(10, 30)):
        x = random.randint(0, IMG_W - 1)
        y = random.randint(0, IMG_H - 1)
        r = random.randint(2, 8)
        color = (
            random.randint(170, 240),
            random.randint(180, 250),
            random.randint(180, 255)
        )
        cv2.circle(overlay, (x, y), r, color, -1, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, random.uniform(0.04, 0.12), out, 1.0, 0)
    return out


def create_face_texture(digit: str, size=256):
    tex = np.zeros((size, size, 3), dtype=np.uint8)

    outer_color = (
        random.randint(0, 15),
        random.randint(210, 235),
        random.randint(235, 255)
    )

    inner_color = (
        random.randint(0, 15),
        random.randint(170, 205),
        random.randint(190, 230)
    )

    tex[:] = outer_color

    margin = random.randint(28, 42)
    cv2.rectangle(
        tex,
        (margin, margin),
        (size - margin, size - margin),
        inner_color,
        -1,
        cv2.LINE_AA
    )

    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX
    ])

    font_scale = random.uniform(4.0, 5.0)
    thickness = random.randint(9, 14)

    text_size, _ = cv2.getTextSize(digit, font, font_scale, thickness)
    tw, th = text_size

    tx = (size - tw) // 2 + random.randint(-8, 8)
    ty = (size + th) // 2 + random.randint(-8, 8)

    cv2.putText(
        tex,
        digit,
        (tx, ty),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    if random.random() < 0.4:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        tex = cv2.warpAffine(
            tex,
            M,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderValue=outer_color
        )

    return tex


def warp_texture_onto_quad(img, texture, quad):
    h, w = img.shape[:2]
    th, tw = texture.shape[:2]

    src = np.float32([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1]
    ])

    dst = np.float32(quad)

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(texture, M, (w, h), flags=cv2.INTER_LINEAR)

    mask = np.ones((th, tw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_LINEAR)

    mask_bool = warped_mask > 0
    img[mask_bool] = warped[mask_bool]

    return img


def draw_shadow(img, projected_pts):
    pts = np.array(projected_pts, dtype=np.float32)

    cx = int(np.mean(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))

    width = int(max(25, np.max(pts[:, 0]) - np.min(pts[:, 0])) * 0.35)
    height = int(max(10, width * 0.25))

    overlay = img.copy()

    cv2.ellipse(
        overlay,
        (cx, min(IMG_H - 5, y_max + 12)),
        (width, height),
        0,
        0,
        360,
        (150, 150, 150),
        -1,
        cv2.LINE_AA
    )

    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    return img


CUBE_VERTS = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=np.float32)


FACES = {
    "front":  [0, 3, 2, 1],
    "back":   [4, 5, 6, 7],
    "left":   [0, 4, 7, 3],
    "right":  [1, 2, 6, 5],
    "top":    [3, 7, 6, 2],
    "bottom": [0, 1, 5, 4],
}


def visible_faces(cam_pts):
    visible = []

    for name, idxs in FACES.items():
        face_pts = cam_pts[idxs]
        p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]

        normal = np.cross(p1 - p0, p2 - p0)

        if np.dot(normal, p0) < 0:
            mean_z = np.mean(face_pts[:, 2])
            visible.append((name, idxs, mean_z))

    visible.sort(key=lambda x: x[2], reverse=True)
    return visible


def generate_valid_geometry():
    for _ in range(MAX_RETRIES):
        f = random.uniform(720, 900)

        cx = IMG_W / 2 + random.uniform(-20, 20)
        cy = IMG_H / 2 + random.uniform(-15, 15)

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Более безопасные углы: ракурсы разные, но без диких искажений
        rx = np.deg2rad(random.uniform(-25, 25))
        ry = np.deg2rad(random.uniform(-42, 42))
        rz = np.deg2rad(random.uniform(-18, 18))

        R = rotation_matrix_xyz(rx, ry, rz)

        scale = random.uniform(1.15, 1.75)

        t = np.array([
            random.uniform(-0.45, 0.45),
            random.uniform(-0.30, 0.30),
            random.uniform(4.7, 6.3)
        ], dtype=np.float32)

        obj_pts = CUBE_VERTS * scale
        cam_pts = (R @ obj_pts.T).T + t

        if np.any(cam_pts[:, 2] <= 1.0):
            continue

        proj_pts = project_points(cam_pts, K)
        visible = visible_faces(cam_pts)

        if is_good_cube(proj_pts, visible):
            return proj_pts, visible

    return None, None


def generate_cube_image(digit="1", underwater=False):
    img = create_background(underwater=underwater)

    proj_pts, visible = generate_valid_geometry()

    if proj_pts is None:
        # запасной вариант: если 100 раз не вышло, пробуем заново
        return generate_cube_image(digit, underwater)

    img = draw_shadow(img, proj_pts)

    for face_name, idxs, _ in visible:
        texture = create_face_texture(digit, size=256)

        quad_2d = order_quad_tl_tr_br_bl(proj_pts[idxs])

        img = warp_texture_onto_quad(img, texture, quad_2d)

        cv2.polylines(
            img,
            [quad_2d.astype(np.int32)],
            True,
            (0, 180, 220),
            random.randint(1, 2),
            cv2.LINE_AA
        )

    alpha = random.uniform(0.92, 1.12)
    beta = random.randint(-8, 12)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if underwater:
        img = add_underwater_effect(img)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="dataset_digits")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--digits", type=str, default="1-9")
    parser.add_argument("--underwater", type=int, default=0)

    args = parser.parse_args()

    out_root = Path(args.out)
    digits = parse_digits(args.digits)

    if not digits:
        raise ValueError("Не удалось распарсить digits")

    for digit in digits:
        digit_dir = out_root / digit
        digit_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.count):
            img = generate_cube_image(
                digit=digit,
                underwater=bool(args.underwater)
            )

            out_path = digit_dir / f"cube_{digit}_{i:05d}.jpg"

            cv2.imwrite(
                str(out_path),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(88, 96)]
            )

        print(f"Готово: цифра {digit}, изображений: {args.count}")

    print(f"\nСохранено в: {out_root.resolve()}")


if __name__ == "__main__":
    main()