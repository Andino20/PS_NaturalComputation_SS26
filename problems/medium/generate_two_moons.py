import csv
import math
import random
import struct
import zlib
from pathlib import Path


def generate_two_moons(n_samples=1000, noise=0.2, random_state=42):
    rng = random.Random(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    points = []

    for i in range(n_outer):
        theta = math.pi * i / (n_outer - 1)
        x = math.cos(theta) + rng.gauss(0, noise)
        y = math.sin(theta) + rng.gauss(0, noise)
        points.append((x, y, 0))

    for i in range(n_inner):
        theta = math.pi * i / (n_inner - 1)
        x = 1 - math.cos(theta) + rng.gauss(0, noise)
        y = 1 - math.sin(theta) - 0.5 + rng.gauss(0, noise)
        points.append((x, y, 1))

    rng.shuffle(points)
    return points


def stratified_train_test_split(points, test_size=0.2, random_state=42):
    rng = random.Random(random_state)
    by_label = {}

    for point in points:
        by_label.setdefault(point[2], []).append(point)

    train = []
    test = []
    for label_points in by_label.values():
        rng.shuffle(label_points)
        n_test = round(len(label_points) * test_size)
        test.extend(label_points[:n_test])
        train.extend(label_points[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(path, points):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["x", "y", "label"])
        writer.writerows(points)


def write_png(path, pixels, width, height):
    def chunk(chunk_type, data):
        body = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row_start = y * width * 3
        raw.extend(pixels[row_start : row_start + width * 3])

    data = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(data)


def save_plot(path, points, width=800, height=800):
    pixels = bytearray([255, 255, 255] * width * height)
    padding = 70
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    min_x, max_x = cx - span / 2, cx + span / 2
    min_y, max_y = cy - span / 2, cy + span / 2

    def set_pixel(px, py, color):
        if 0 <= px < width and 0 <= py < height:
            offset = (py * width + px) * 3
            pixels[offset : offset + 3] = bytes(color)

    def draw_line(x1, y1, x2, y2, color):
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            set_pixel(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def to_pixel(x, y):
        px = padding + round((x - min_x) / (max_x - min_x) * (width - 2 * padding))
        py = height - padding - round((y - min_y) / (max_y - min_y) * (height - 2 * padding))
        return px, py

    for step in range(11):
        gx = padding + round(step / 10 * (width - 2 * padding))
        gy = padding + round(step / 10 * (height - 2 * padding))
        draw_line(gx, padding, gx, height - padding, (225, 225, 225))
        draw_line(padding, gy, width - padding, gy, (225, 225, 225))

    draw_line(padding, padding, padding, height - padding, (80, 80, 80))
    draw_line(padding, height - padding, width - padding, height - padding, (80, 80, 80))

    colors = {0: (52, 116, 204), 1: (220, 67, 67)}
    for x, y, label in points:
        px, py = to_pixel(x, y)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                distance = dx * dx + dy * dy
                if distance <= 16:
                    set_pixel(px + dx, py + dy, colors[label])
                elif distance <= 25:
                    set_pixel(px + dx, py + dy, (35, 35, 35))

    write_png(path, pixels, width, height)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent

    data = generate_two_moons(n_samples=1000, noise=0.2, random_state=42)
    train_data, test_data = stratified_train_test_split(data, test_size=0.2, random_state=42)

    write_csv(output_dir / "two_moons_train.csv", train_data)
    write_csv(output_dir / "two_moons_test.csv", test_data)
    save_plot(output_dir / "two_moons.png", data)
