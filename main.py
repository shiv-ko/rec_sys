# c:\Users\kouki\lab\rec_sys\main.py
# MovieLens 100K のCF実験をまとめて実行するスクリプト。
# 目的: 教材の比較実験を手早く再現し、結果をレポートへ反映するため。
# Relevant: report.md, plan.md, dev.md
"""End-to-end MovieLens 100K collaborative filtering experiment runner."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RatingMetrics:
    """Container for rating prediction metrics."""

    rmse: float
    mae: float
    # ※ RMSE/MAE の 2 指標をまとめて扱うための軽量なデータ入れ物。


def load_movielens_100k(data_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Load MovieLens 100K ratings and encode ids to zero-based indices.

    The original user/item ids start from 1 but may be non-contiguous. Factorizing
    creates consecutive 0-based `user_idx` and `item_idx` columns that are easier
    to use as array indices.
    元のユーザー/アイテムIDは1から始まるが、連続していない場合がある。factorizeすることで、
    連続した0ベースの`user_idx`と`item_idx`列が作成され、配列のインデックスとして
    使いやすくなる。
    """

    # 1) MovieLens 100K はタブ区切りかつヘッダ無しなので明示的に列名を与えつつ読み込む。
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    # 2) 元の ID は欠番を含むため、pd.factorize で連番インデックスを生成し密な 0 始まりの ID に変換。
    df["user_idx"], user_index = pd.factorize(df["user_id"], sort=True)
    df["item_idx"], item_index = pd.factorize(df["item_id"], sort=True)
    # 3) factorize が返すユニーク配列の長さがユーザ数・アイテム数になる。
    num_users = len(user_index)
    num_items = len(item_index)
    return df, num_users, num_items


def leave_one_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply per-user leave-one-out split using timestamps to mimic future prediction."""

    # ここでは「ユーザごとの最新 1 本を未来の評価と見なす」ので、まずは (user_idx, timestamp) で昇順ソートする。
    sorted_df = ratings.sort_values(["user_idx", "timestamp"])  # chronological order
    sorted_df = ratings.sort_values(["user_idx", "timestamp"])  # 時系列順
    # groupby(...).tail(1) によってユーザごとの最後の 1 レコードを取り出し、テストインデックスとして確保する。
    test_idx = sorted_df.groupby("user_idx", group_keys=False).tail(1).index
    # テスト = 未来 1 件、トレイン = それ以前の履歴という推薦タスクらしい分割が得られる。
    test_df = sorted_df.loc[test_idx]
    train_df = sorted_df.drop(test_idx)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_interaction_matrix(
    train_df: pd.DataFrame, num_users: int, num_items: int
) -> np.ndarray:
    """Create dense user-item matrix where 0 encodes 'unobserved' interactions."""

    R = np.zeros((num_users, num_items), dtype=np.float32)
    # 未観測 = 0 を用い、観測済みセルのみ評価値を代入する（0 は本当の評価ではなく「欠損」のマーカー）。
    # 後続の CF 計算では R > 0 かどうかで観測判定を行うため、符号付きで保持しておく。
    R[train_df["user_idx"], train_df["item_idx"]] = train_df["rating"].astype(np.float32)
    return R


def build_interaction_triplets(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert training dataframe to aligned (u, i, r) arrays for SGD."""

    users = train_df["user_idx"].to_numpy(dtype=np.int32)
    items = train_df["item_idx"].to_numpy(dtype=np.int32)
    ratings = train_df["rating"].to_numpy(dtype=np.float32)
    return users, items, ratings


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""

    # 2 乗誤差の平均を取って平方根を戻すことで、大きな外れ値を強く罰する評価指標になる。
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""

    # 絶対値誤差の平均は外れ値に頑健で、直感的な「何点ずれたか」を示す補助指標。
    return float(np.mean(np.abs(y_true - y_pred)))


def hit_rate_at_k(
    model: "BaseRecommender",
    R_train: np.ndarray,
    test_df: pd.DataFrame,
    k: int,
) -> float:
    """Compute HitRate@K under leave-one-out by ranking unseen items per user."""

    num_items = R_train.shape[1]
    # leave-one-out なので各ユーザにつき 1 アイテムをランキング対象にする。
    hits = 0
    for row in test_df.itertuples(index=False):
        user = int(row.user_idx)
        target_item = int(row.item_idx)
        # ① そのユーザが既に評価したアイテムは推薦候補から除外する。
        seen_mask = R_train[user] > 0
        candidate_items = np.where(~seen_mask)[0]
        if candidate_items.size == 0:
            continue
        # 全未観測アイテムに対してスコアを出し、トップ K に入ったかどうかを見る。
        scores = np.array([model.predict_single(user, item) for item in candidate_items])
        # 候補アイテム数がkより多い場合は、高速なargpartitionで上位k件のインデックスを取得する。
        if candidate_items.size > k:
            top_indices = np.argpartition(-scores, k - 1)[:k]
        else:
            top_indices = np.arange(candidate_items.size)
        # 上位k件のアイテムIDのセットを作成する。
        top_items = set(candidate_items[top_indices])
        if target_item in top_items:
            hits += 1
    return hits / len(test_df)


class BaseRecommender:
    """Interface for recommender models."""

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - インターフェースのみ
        raise NotImplementedError

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError


class ItemMeanBaseline(BaseRecommender):
    """Simple baseline that returns item means for sanity-check comparisons."""
    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.item_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, num_items: int) -> None:
        """Compute per-item and global means on the training set."""

        self.global_mean = float(train_df["rating"].mean())
        sums = np.zeros(num_items, dtype=np.float32)
        counts = np.zeros(num_items, dtype=np.int32)
        # 各アイテムごとに (総和, 件数) を逐次加算し、最終的に平均を得る。
        for row in train_df.itertuples(index=False):
            sums[int(row.item_idx)] += float(row.rating)
            counts[int(row.item_idx)] += 1
        self.item_means = np.full(num_items, self.global_mean, dtype=np.float32)
        valid_mask = counts > 0
        # Avoid dividing by zero for items never rated in the training split.
        # 学習データで一度も評価されなかったアイテムによるゼロ除算を回避する。
        self.item_means[valid_mask] = sums[valid_mask] / counts[valid_mask]

    def predict_single(self, user_idx: int, item_idx: int) -> float:  # pylint: disable=unused-argument
        if self.item_means is None:
            raise ValueError("Model must be fitted before prediction.")
        # 固定のアイテム平均値を返すだけなので、ユーザ情報は使用しない。
        return float(self.item_means[item_idx])


class UserBasedCF(BaseRecommender):
    """User-based CF with mean-centering and cosine similarity."""

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.R_train: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None
        self.user_sim: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(self, R_train: np.ndarray) -> None:
        """Pre-compute user means, centered matrix, and user-user similarities."""

        self.R_train = R_train
        mask = R_train > 0
        counts = mask.sum(axis=1)
        sums = (R_train * mask).sum(axis=1)
        self.global_mean = float(sums.sum() / max(1, counts.sum()))
        self.user_means = np.where(counts > 0, sums / counts, self.global_mean)
        # 楽観/悲観バイアスを除去するため、観測セルのみ mean-centering する。
        centered = np.where(mask, R_train - self.user_means[:, None], 0.0)
        # コサイン類似度計算の準備として、ユーザごとのベクトルノルムを計算。
        norms = np.linalg.norm(centered, axis=1)
        denom = norms[:, None] * norms[None, :]
        denom[denom == 0] = 1e-8  # guard against zero vectors
        denom[denom == 0] = 1e-8  # ゼロベクトル対策
        # コサイン類似度 = 中心化行列の内積 ÷ ノルム積。
        sim = centered @ centered.T / denom
        np.fill_diagonal(sim, 0.0)
        self.user_sim = sim

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        if self.R_train is None or self.user_means is None or self.user_sim is None:
            raise ValueError("Model must be fitted before prediction.")
        if self.R_train[user_idx, item_idx] > 0:
            return float(self.R_train[user_idx, item_idx])
        # アイテム i を評価済みのユーザのみが投票候補になる。
        candidate_users = np.where(self.R_train[:, item_idx] > 0)[0]
        if candidate_users.size == 0:
            return float(self.user_means[user_idx])
        sims = self.user_sim[user_idx, candidate_users]
        if np.all(np.abs(sims) == 0):
            return float(self.user_means[user_idx])
        # 絶対値の大きい類似度から上位 k を選択し、符号付きで重み付ける。
        order = np.argsort(-np.abs(sims))[: self.k]
        neighbors = candidate_users[order]
        neighbor_sims = sims[order]
        neighbor_ratings = self.R_train[neighbors, item_idx]
        neighbor_means = self.user_means[neighbors]
        diff = neighbor_ratings - neighbor_means
        # mean-centering 後の差分を類似度で重み付け、最終的に自分の平均に足し戻す。
        numerator = np.sum(neighbor_sims * diff)
        denom = np.sum(np.abs(neighbor_sims))
        if denom == 0:
            return float(self.user_means[user_idx])
        return float(self.user_means[user_idx] + numerator / denom)


class ItemBasedCF(BaseRecommender):
    """Item-based CF with adjusted cosine similarity (user-centered ratings)."""

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.R_train: Optional[np.ndarray] = None
        self.item_sim: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(self, R_train: np.ndarray) -> None:
        """Pre-compute adjusted cosine item-item similarities."""

        self.R_train = R_train
        mask = R_train > 0
        counts = mask.sum(axis=1)
        sums = (R_train * mask).sum(axis=1)
        self.global_mean = float(sums.sum() / max(1, counts.sum()))
        self.user_means = np.where(counts > 0, sums / counts, self.global_mean)
        # Adjusted Cosine: ユーザ平均差分を消してからアイテム同士のコサイン類似度を取る。
        centered = np.where(mask, R_train - self.user_means[:, None], 0.0)
        item_profiles = centered.T
        # ユーザベクトルではなくアイテムベクトル側でノルムを計算する点が user-based との違い。
        norms = np.linalg.norm(item_profiles, axis=1)
        denom = norms[:, None] * norms[None, :]
        denom[denom == 0] = 1e-8
        sim = item_profiles @ item_profiles.T / denom
        np.fill_diagonal(sim, 0.0)
        self.item_sim = sim

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        if self.R_train is None or self.item_sim is None or self.user_means is None:
            raise ValueError("Model must be fitted before prediction.")
        if self.R_train[user_idx, item_idx] > 0:
            return float(self.R_train[user_idx, item_idx])
        # そのユーザが評価済みのアイテム集合のみが類似アイテム候補。
        user_history = np.where(self.R_train[user_idx] > 0)[0]
        if user_history.size == 0:
            return float(self.user_means[user_idx]) if not np.isnan(self.user_means[user_idx]) else self.global_mean
        sims = self.item_sim[item_idx, user_history]
        if np.all(np.abs(sims) == 0):
            return float(self.user_means[user_idx])
        # 類似度の絶対値でソートし、Top-k の加重平均をとる。
        order = np.argsort(-np.abs(sims))[: self.k]
        neighbors = user_history[order]
        weights = sims[order]
        neighbor_ratings = self.R_train[user_idx, neighbors]
        # 調和が取れない場合に分母が 0 にならないよう、絶対値和で正規化して平均を作る。
        numerator = np.sum(weights * neighbor_ratings)
        denom = np.sum(np.abs(weights))
        if denom == 0:
            return float(self.user_means[user_idx])
        return float(numerator / denom)


class MFRecommender(BaseRecommender):
    """Simple matrix factorization (SGD) with user/item biases."""

    def __init__(
        self,
        factors: int = 16,
        epochs: int = 3,
        lr: float = 0.01,
        reg: float = 0.02,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.num_users: int = 0
        self.num_items: int = 0

    def fit(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
    ) -> None:
        """Train latent factors with plain SGD over observed triples."""

        self.num_users = num_users
        self.num_items = num_items
        rng = np.random.default_rng(self.random_state)
        self.user_factors = 0.1 * rng.standard_normal((num_users, self.factors), dtype=np.float32)
        self.item_factors = 0.1 * rng.standard_normal((num_items, self.factors), dtype=np.float32)
        self.user_bias = np.zeros(num_users, dtype=np.float32)
        self.item_bias = np.zeros(num_items, dtype=np.float32)
        users, items, ratings = build_interaction_triplets(train_df)
        self.global_mean = float(ratings.mean())

        for epoch in range(self.epochs):
            order = rng.permutation(len(ratings))
            for idx in order:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]
                pred = self.predict_single(u, i)
                err = r - pred
                # SGD update (bias + latent factors)
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                pu = self.user_factors[u]
                qi = self.item_factors[i]
                grad_pu = err * qi - self.reg * pu
                grad_qi = err * pu - self.reg * qi
                self.user_factors[u] += self.lr * grad_pu
                self.item_factors[i] += self.lr * grad_qi
            if self.verbose:
                train_pred = np.array([self.predict_single(u, i) for u, i in zip(users, items)], dtype=np.float32)
                train_rmse = rmse(ratings, train_pred)
                print(f"[MF] epoch={epoch+1}/{self.epochs}, train_rmse={train_rmse:.4f}")

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        if any(
            comp is None
            for comp in (
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias,
            )
        ):
            raise ValueError("Model must be fitted before prediction.")
        user_idx = int(user_idx)
        item_idx = int(item_idx)
        if user_idx >= self.num_users or item_idx >= self.num_items:
            return self.global_mean
        return float(
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )


def evaluate_model(
    model: BaseRecommender,
    test_df: pd.DataFrame,
) -> RatingMetrics:
    """Evaluate a model on the leave-one-out test set."""

    # leave-one-out なので単純に 943 件すべての予測値を並べて指標計算する。
    preds = np.array([model.predict_single(r.user_idx, r.item_idx) for r in test_df.itertuples(index=False)])
    truth = test_df["rating"].to_numpy(dtype=np.float32)
    return RatingMetrics(rmse=rmse(truth, preds), mae=mae(truth, preds))


def run_experiments(
    data_path: Path,
    k_values: Iterable[int],
    compute_hit_rate: bool,
    mf_factors: Iterable[int],
    mf_epochs: int,
    mf_lr: float,
    mf_reg: float,
    mf_verbose: bool,
) -> None:
    """Execute full pipeline from data loading to reporting."""

    ratings, num_users, num_items = load_movielens_100k(data_path)
    train_df, test_df = leave_one_out_split(ratings)
    R_train = build_interaction_matrix(train_df, num_users, num_items)

    baseline = ItemMeanBaseline()
    baseline.fit(train_df, num_items)
    baseline_metrics = evaluate_model(baseline, test_df)

    user_cf_results: List[Dict[str, float]] = []
    for k in k_values:
        model = UserBasedCF(k=k)
        model.fit(R_train)
        metrics = evaluate_model(model, test_df)
        user_cf_results.append({"k": k, "rmse": metrics.rmse, "mae": metrics.mae, "model": "UserCF", "instance": model})

    item_cf_results: List[Dict[str, float]] = []
    for k in k_values:
        model = ItemBasedCF(k=k)
        model.fit(R_train)
        metrics = evaluate_model(model, test_df)
        item_cf_results.append({"k": k, "rmse": metrics.rmse, "mae": metrics.mae, "model": "ItemCF", "instance": model})

    mf_results: List[Dict[str, float]] = []
    for f in mf_factors:
        model = MFRecommender(factors=f, epochs=mf_epochs, lr=mf_lr, reg=mf_reg, verbose=mf_verbose)
        model.fit(train_df, num_users, num_items)
        metrics = evaluate_model(model, test_df)
        mf_results.append({"factors": f, "epochs": mf_epochs, "lr": mf_lr, "reg": mf_reg, "rmse": metrics.rmse, "mae": metrics.mae, "model": "MF", "instance": model})

    user_df = pd.DataFrame([{k: v for k, v in row.items() if k != "instance"} for row in user_cf_results])
    item_df = pd.DataFrame([{k: v for k, v in row.items() if k != "instance"} for row in item_cf_results])
    mf_df = pd.DataFrame([{k: v for k, v in row.items() if k != "instance"} for row in mf_results])

    best_user_idx = user_df["rmse"].idxmin()
    best_item_idx = item_df["rmse"].idxmin()
    best_mf_idx = mf_df["rmse"].idxmin()
    best_user_row = user_cf_results[int(best_user_idx)]
    best_item_row = item_cf_results[int(best_item_idx)]
    best_mf_row = mf_results[int(best_mf_idx)]

    best_user_model = best_user_row["instance"]
    best_item_model = best_item_row["instance"]
    best_mf_model = best_mf_row["instance"]

    hit_rate_k = 10
    baseline_hit = user_hit = item_hit = mf_hit = np.nan
    if compute_hit_rate:
        baseline_hit = hit_rate_at_k(baseline, R_train, test_df, hit_rate_k)
        user_hit = hit_rate_at_k(best_user_model, R_train, test_df, hit_rate_k)
        item_hit = hit_rate_at_k(best_item_model, R_train, test_df, hit_rate_k)
        mf_hit = hit_rate_at_k(best_mf_model, R_train, test_df, hit_rate_k)

    summary_rows = [
        {"model": "ItemMeanBaseline", "k": None, "factors": None, "rmse": baseline_metrics.rmse, "mae": baseline_metrics.mae, "hit_rate@10": baseline_hit},
        {"model": f"UserCF(k={best_user_row['k']})", "k": best_user_row["k"], "factors": None, "rmse": best_user_row["rmse"], "mae": best_user_row["mae"], "hit_rate@10": user_hit},
        {"model": f"ItemCF(k={best_item_row['k']})", "k": best_item_row["k"], "factors": None, "rmse": best_item_row["rmse"], "mae": best_item_row["mae"], "hit_rate@10": item_hit},
        {"model": f"MF(f={best_mf_row['factors']},e={best_mf_row['epochs']})", "k": None, "factors": best_mf_row["factors"], "rmse": best_mf_row["rmse"], "mae": best_mf_row["mae"], "hit_rate@10": mf_hit},
    ]
    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Train/Test Split Stats ===")
    print(f"Train ratings: {len(train_df):,}, Test ratings: {len(test_df):,}")
    print(f"Users: {num_users}, Items: {num_items}")

    print("\n=== Baseline vs CF Summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== UserCF k-sweep (lower RMSE/MAE is better) ===")
    print(user_df.to_string(index=False))

    print("\n=== ItemCF k-sweep (lower RMSE/MAE is better) ===")
    print(item_df.to_string(index=False))

    print("\n=== MF factor sweep (lower RMSE/MAE is better) ===")
    print(mf_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens 100K CF comparison")
    # 実行時オプションは (i) データパス (ii) 近傍数リスト (iii) HitRate を省略するか、の 3 種類。
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ml-100k/u.data"),
        help="Path to MovieLens 100K u.data file",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40],
        help="Neighbor counts to sweep for CF models",
    )
    parser.add_argument(
        "--mf-factors",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Latent dimensions to sweep for matrix factorization",
    )
    parser.add_argument(
        "--mf-epochs",
        type=int,
        default=3,
        help="Epochs for matrix factorization SGD",
    )
    parser.add_argument(
        "--mf-lr",
        type=float,
        default=0.01,
        help="Learning rate for matrix factorization",
    )
    parser.add_argument(
        "--mf-reg",
        type=float,
        default=0.02,
        help="L2 regularization strength for matrix factorization",
    )
    parser.add_argument(
        "--mf-verbose",
        action="store_true",
        help="Print MF train RMSE per epoch",
    )
    parser.add_argument(
        "--skip-hit-rate",
        action="store_true",
        help="Skip HitRate@10 computation to save time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        args.data_path,
        args.k_values,
        compute_hit_rate=not args.skip_hit_rate,
        mf_factors=args.mf_factors,
        mf_epochs=args.mf_epochs,
        mf_lr=args.mf_lr,
        mf_reg=args.mf_reg,
        mf_verbose=args.mf_verbose,
    )


if __name__ == "__main__":
    # コマンドラインから実行された場合のみフルパイプラインを回す。
    main()
