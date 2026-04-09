import argparse
import pickle
from pathlib import Path
import numpy as np
from train import _Scaler, TPSMapper


def load_model(path: Path):
    with open(path, "rb") as f:
        s = pickle.load(f)

    model = TPSMapper(regularization=s["lam"])
    model.ctrl_pts = s["ctrl_pts"]
    model.w_x = s["w_x"]
    model.w_y = s["w_y"]
    model._scaler_src = s["scaler_src"]
    model._scaler_dst = s["scaler_dst"]

    return model


def main():
    parser = argparse.ArgumentParser(description="тест TPS артефакта")
    parser.add_argument("--artifacts", required=True, help="путь к папке artifacts")
    parser.add_argument("--source", required=True, choices=["top", "bottom"])
    parser.add_argument("--x_y", type=float, nargs=2, required=True, help="x y пиксель в source-камере через пробел")
    args = parser.parse_args()

    model_path = Path(args.artifacts) / args.source / "tps.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model(model_path)
    x, y = args.x_y
    pt = np.array([[x, y]], dtype=np.float64)  # модель ждёт [N,2], поэтому оборачиваем в батч из 1 точки
    pred = model.predict(pt)[0]

    print("\nsource:", args.source)
    print("input:  ", (x, y))
    print("door2:  ", (round(pred[0], 2), round(pred[1], 2)))


if __name__ == "__main__":
    main()