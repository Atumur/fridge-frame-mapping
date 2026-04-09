import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from train import _Scaler, TPSMapper


ARTIFACTS_LOCKED_PATH = Path("./artifacts")


def load_model(source):
    model_path = ARTIFACTS_LOCKED_PATH / source / "tps.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with open(model_path, "rb") as f:
        s = pickle.load(f)

    model = TPSMapper(regularization=s["lam"])
    model.ctrl_pts = s["ctrl_pts"]
    model.w_x = s["w_x"]
    model.w_y = s["w_y"]
    model._scaler_src = s["scaler_src"]
    model._scaler_dst = s["scaler_dst"]

    return model


def extract_frame_number(path):
    m = re.search(r"frame_(\d+)", path.name)
    if not m:
        raise ValueError(f"не удалось вытащить номер кадра из {path.name}")
    return int(m.group(1))


def find_closest_frame(door2_dir, src_frame_num):
    # собираем все кадры в door2 и ищем минимальную разницу по номеру тк они не одноименные
    candidates = []

    for p in door2_dir.glob("frame_*.jpg"):
        try:
            num = extract_frame_number(p)
            candidates.append((abs(num - src_frame_num), p))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(f"в {door2_dir} нет frame_*.jpg")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def pick_point(img):
    plt.imshow(img)
    plt.title("клик точку на изображении")
    pts = plt.ginput(1)
    plt.close()
    return pts[0]


def draw_point(img, point, title):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(point[0], point[1], s=120)
    ax.set_title(title)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="путь до bottom/top кадра")
    args = parser.parse_args()

    src_path = Path(args.image)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    source = src_path.parent.name  # bottom или top
    session_dir = src_path.parents[1]
    door2_dir = session_dir / "door2"

    src_frame_num = extract_frame_number(src_path)
    door2_path = find_closest_frame(door2_dir, src_frame_num)

    print("source image:", src_path)
    print("closest door2 image:", door2_path)

    model = load_model(source)

    src_img = Image.open(src_path)
    door2_img = Image.open(door2_path)

    x, y = pick_point(src_img)
    print("picked:", x, y)

    pt = np.array([[x, y]], dtype=np.float64)
    pred = model.predict(pt)[0]

    print("predicted door2:", pred)

    draw_point(door2_img, pred, "предсказанная точка в door2")


if __name__ == "__main__":
    main()