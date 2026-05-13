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


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}


def text_width(text, scale):
    return max(0, (len(text) * 6 - 1) * scale)


def save_plot(path, points, width=800, height=800):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        save_basic_plot(path, points, width, height)
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        c=[point[2] for point in points],
        cmap="Spectral",
        edgecolors="k",
        alpha=0.7,
    )
    plt.title("Two Moons")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(path)
    plt.close()


def save_basic_plot(path, points, width=800, height=800):
    pixels = bytearray([255, 255, 255] * width * height)
    left = 100
    right = 80
    top = 96
    bottom = 88
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    min_x, max_x = cx - span / 2, cx + span / 2
    min_y, max_y = cy - span / 2, cy + span / 2
    plot_width = width - left - right
    plot_height = height - top - bottom

    def set_pixel(px, py, color):
        if 0 <= px < width and 0 <= py < height:
            offset = (py * width + px) * 3
            pixels[offset : offset + 3] = bytes(color)

    def draw_text(x, y, text, color=(0, 0, 0), scale=2):
        cursor_x = x
        for char in text.upper():
            pattern = FONT.get(char, FONT[" "])
            for row_index, row in enumerate(pattern):
                for col_index, value in enumerate(row):
                    if value == "1":
                        for sy in range(scale):
                            for sx in range(scale):
                                set_pixel(
                                    cursor_x + col_index * scale + sx,
                                    y + row_index * scale + sy,
                                    color,
                                )
            cursor_x += 6 * scale

    def draw_text_rotated_left(x, y, text, color=(0, 0, 0), scale=2):
        total_width = text_width(text, scale)
        cursor_units = 0
        for char in text.upper():
            pattern = FONT.get(char, FONT[" "])
            for row_index, row in enumerate(pattern):
                for col_index, value in enumerate(row):
                    if value == "1":
                        unit_x = cursor_units + col_index
                        unit_y = row_index
                        for sy in range(scale):
                            for sx in range(scale):
                                source_x = unit_x * scale + sx
                                source_y = unit_y * scale + sy
                                set_pixel(x + source_y, y + total_width - 1 - source_x, color)
            cursor_units += 6

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

    def draw_dashed_line(x1, y1, x2, y2, color, dash=6, gap=4):
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            set_pixel(x1, y1, color)
            return

        for step in range(steps + 1):
            if step % (dash + gap) < dash:
                px = round(x1 + dx * step / steps)
                py = round(y1 + dy * step / steps)
                set_pixel(px, py, color)

    def to_pixel(x, y):
        px = left + round((x - min_x) / (max_x - min_x) * plot_width)
        py = height - bottom - round((y - min_y) / (max_y - min_y) * plot_height)
        return px, py

    for step in range(11):
        gx = left + round(step / 10 * plot_width)
        gy = top + round(step / 10 * plot_height)
        draw_dashed_line(gx, top, gx, height - bottom, (205, 205, 205))
        draw_dashed_line(left, gy, width - right, gy, (205, 205, 205))

    draw_line(left, top, width - right, top, (0, 0, 0))
    draw_line(left, height - bottom, width - right, height - bottom, (0, 0, 0))
    draw_line(left, top, left, height - bottom, (0, 0, 0))
    draw_line(width - right, top, width - right, height - bottom, (0, 0, 0))

    title = "Two Moons"
    x_label = "Feature X"
    y_label = "Feature Y"
    draw_text((width - text_width(title, 3)) // 2, 74, title, scale=3)
    draw_text((width - text_width(x_label, 2)) // 2, 746, x_label, scale=2)
    draw_text_rotated_left(32, (height - text_width(y_label, 2)) // 2, y_label, scale=2)

    colors = {0: (124, 117, 182), 1: (194, 59, 114)}
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
