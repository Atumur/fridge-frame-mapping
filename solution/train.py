from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


def load_pairs_from_file(json_path):
    # читает coords_top.json или coords_bottom.json. и возвращает список пар (src_pts, dst_pts), где
    #   src_pts - координаты на камере top/bottom  [N, 2]
    #   dst_pts - координаты на камере door2       [N, 2]
    if not json_path.exists():
        return []

    with open(json_path) as f:
        records = json.load(f)

    pairs = []
    for rec in records:
        img1_coords = {p["number"]: (p["x"], p["y"]) for p in rec["image1_coordinates"]}
        img2_coords = {p["number"]: (p["x"], p["y"]) for p in rec["image2_coordinates"]}

        common = sorted(set(img1_coords) & set(img2_coords))
        if len(common) < 4:
            continue

        # image1 = door2, image2 = top/bottom
        dst_pts = np.array([img1_coords[n] for n in common], dtype=np.float64)
        src_pts = np.array([img2_coords[n] for n in common], dtype=np.float64)
        pairs.append((src_pts, dst_pts))

    return pairs


def load_split(data_root, split, source):
    #собирает все точки соответствия из заданного сплита и возвращает X [N,2] (source-камера) и y [N,2] (door2)
    with open(data_root / "split.json") as f:
        session_paths = json.load(f)[split]

    all_src, all_dst = [], []
    for rel_path in session_paths:
        coord_file = data_root / rel_path / f"coords_{source}.json"
        for src_pts, dst_pts in load_pairs_from_file(coord_file):
            all_src.append(src_pts)
            all_dst.append(dst_pts)

    if not all_src:
        raise ValueError(f"Нет данных для split='{split}', source='{source}'")

    return np.vstack(all_src), np.vstack(all_dst)


class _Scaler:
    # MinMax scaler по обеим осям независимо

    def fit(self, pts: np.ndarray):
        self.min_ = pts.min(axis=0)
        self.range_ = pts.max(axis=0) - pts.min(axis=0)
        self.range_ = np.where(self.range_ > 0, self.range_, 1.0)
        return self

    def transform(self, pts: np.ndarray):
        return (pts - self.min_) / self.range_

    def inverse_transform(self, pts: np.ndarray):
        return pts * self.range_ + self.min_


class TPSMapper:
    def __init__(self, regularization: float = 1e-3):
        self.lam = regularization
        self.ctrl_pts: np.ndarray | None = None
        self.w_x: np.ndarray | None = None
        self.w_y: np.ndarray | None = None
        self._scaler_src: _Scaler | None = None
        self._scaler_dst: _Scaler | None = None

    @staticmethod
    def _kernel(r2: np.ndarray):
        # U(r) = r^2·log(r^2), U(0) = 0
        return np.where(r2 > 0, r2 * np.log(r2 + 1e-12), 0.0)

    def _K(self, a: np.ndarray, b: np.ndarray):
        diff = a[:, None, :] - b[None, :, :]
        return self._kernel((diff ** 2).sum(-1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._scaler_src = _Scaler().fit(X)
        self._scaler_dst = _Scaler().fit(y)
        Xn = self._scaler_src.transform(X)
        yn = self._scaler_dst.transform(y)

        M = len(Xn)
        K = self._K(Xn, Xn) + self.lam * np.eye(M)
        P = np.hstack([np.ones((M, 1)), Xn])

        L = np.block([[K, P], [P.T, np.zeros((3, 3))]])

        # lstsq устойчив к почти-вырожденным матрицам (коллинеарные точки и т.п.)
        self.w_x, *_ = np.linalg.lstsq(L, np.concatenate([yn[:, 0], np.zeros(3)]), rcond=None)
        self.w_y, *_ = np.linalg.lstsq(L, np.concatenate([yn[:, 1], np.zeros(3)]), rcond=None)
        self.ctrl_pts = Xn
        return self

    def predict(self, X: np.ndarray):
        assert self.ctrl_pts is not None, "Вызовите fit() сначала."
        Xn = self._scaler_src.transform(X)
        K  = self._K(Xn, self.ctrl_pts)
        P  = np.hstack([np.ones((len(Xn), 1)), Xn])
        M  = len(self.ctrl_pts)

        px = K @ self.w_x[:M] + P @ self.w_x[M:]
        py = K @ self.w_y[:M] + P @ self.w_y[M:]

        return self._scaler_dst.inverse_transform(np.stack([px, py], axis=-1))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "lam":        self.lam,
                "ctrl_pts":   self.ctrl_pts,
                "w_x":        self.w_x,
                "w_y":        self.w_y,
                "scaler_src": self._scaler_src,
                "scaler_dst": self._scaler_dst,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            s = pickle.load(f)
        m = cls(regularization=s["lam"])
        m.ctrl_pts    = s["ctrl_pts"]
        m.w_x         = s["w_x"]
        m.w_y         = s["w_y"]
        m._scaler_src = s["scaler_src"]
        m._scaler_dst = s["scaler_dst"]
        return m


def med(y_true: np.ndarray, y_pred: np.ndarray):
    # Mean Euclidean Distance
    return float(np.sqrt(((y_true - y_pred) ** 2).sum(axis=1)).mean())



def train(data_root, artifacts_dir, lam):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for source in ("top", "bottom"):

        X_train, y_train = load_split(data_root, "train", source)
        X_val,   y_val   = load_split(data_root, "val",   source)
        print(f"  train: {len(X_train):5d} точек   val: {len(X_val):4d} точек")

        t0 = time.time()
        model = TPSMapper(regularization=float(lam))
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_med = med(y_train, model.predict(X_train))
        val_med   = med(y_val,   model.predict(X_val))

        print(f"  lambda={float(lam):.0e}  время={elapsed:.1f}s")
        print(f"  MED train: {train_med:.1f} px   MED val: {val_med:.1f} px")

        save_path = artifacts_dir / source / "tps.pkl"
        model.save(save_path)

        summary[source] = {
            "lam":       float(lam),
            "n_train":   len(X_train),
            "n_val":     len(X_val),
            "med_train": round(train_med, 2),
            "med_val":   round(val_med,   2),
            "elapsed_s": round(elapsed, 1),
        }

    report_path = artifacts_dir / "train_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    for src, m in summary.items():
        print(f"{src:<11s} MED train {m['med_train']:>8.1f} px │MED val {m['med_val']:>7.1f} px")


def main() -> None:
    parser = argparse.ArgumentParser(description="обучение TPS-маппинга top/bottom в door2")
    parser.add_argument("--data", required=True, help="путь к корневой папке, которая содержит split.json, train/, val/")
    parser.add_argument("--artifacts", default="./artifacts",help="куда сохранять модели")
    parser.add_argument("--lam", type=float, default=1e-3, help="Регуляризация TPS λ (default: 1e-3)")
    args = parser.parse_args()

    train(data_root=Path(args.data), artifacts_dir=Path(args.artifacts), lam=args.lam)


if __name__ == "__main__":
    main()
