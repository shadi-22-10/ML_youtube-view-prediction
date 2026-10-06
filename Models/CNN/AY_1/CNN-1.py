#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting in WSL
import matplotlib.pyplot as plt

# ----------------------------
# Args
# ----------------------------
# You can edit these defaults directly in this file 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DEFAULTS = {
    "csv": os.path.normpath(os.path.join(PROJECT_ROOT, "Dataset/Mohamed_Dataset/snapshots/youtube_dataset_filtered.csv")),
    "thumb_dir": os.path.normpath(os.path.join(PROJECT_ROOT, "Dataset/Mohamed_Dataset/thumbnails")),
    "outdir": os.path.normpath(os.path.join(SCRIPT_DIR, "outputs")),
    "img_size": 224,
    "batch_size": 16,
    "epochs_warm": 5,
    "epochs_ft": 15,
    "val_split": 0.30,
    "seed": 42,
}

parser = argparse.ArgumentParser(description="EfficientNetB0: thumbnail -> view prediction (regression)")
parser.add_argument("--csv", default=DEFAULTS["csv"], help="Path to youtube_dataset.csv")
parser.add_argument("--thumb_dir", default=DEFAULTS["thumb_dir"], help="Directory containing thumbnail images")
parser.add_argument("--outdir", default=DEFAULTS["outdir"], help="Output directory")
parser.add_argument("--img_size", type=int, default=DEFAULTS["img_size"], help="Image size (square)")
parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"], help="Batch size (try 12/8 if OOM)")
parser.add_argument("--epochs_warm", type=int, default=DEFAULTS["epochs_warm"], help="Warm-up epochs (EffNet frozen)")
parser.add_argument("--epochs_ft", type=int, default=DEFAULTS["epochs_ft"], help="Fine-tune epochs (top unfrozen)")
parser.add_argument("--val_split", type=float, default=DEFAULTS["val_split"], help="Val+Test split from full data")
parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed")
args = parser.parse_args()

# Ensure output directories exist
os.makedirs(args.outdir, exist_ok=True)
ckpt_dir = os.path.join(args.outdir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# ----------------------------
# TensorFlow / GPU
# ----------------------------
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

# Reproducibility (best-effort)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)

# Mixed precision & XLA
mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

# Be nice to VRAM
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("CSV:", args.csv)
print("Thumb dir:", args.thumb_dir)
print("Out dir:", args.outdir)

# ----------------------------
# Load & prepare data
# ----------------------------
df = pd.read_csv(args.csv, low_memory=False)

required = ["thumbnail_file", "viewCount"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in CSV (found: {df.columns.tolist()})")

# Core filters
df = df[df["thumbnail_file"].notna() & (df["thumbnail_file"] != "#NAME?") & df["viewCount"].notna()]
df["viewCount"] = df["viewCount"].astype(float)

# Optional: drop likely Shorts if columns exist
if args.drop_shorts:
    title_col = "title" if "title" in df.columns else None
    dur_col   = "duration_sec" if "duration_sec" in df.columns else None
    before = len(df)
    if title_col:
        mask_title = ~df[title_col].fillna("").str.lower().str.contains(r"#shorts|#fyp")
        df = df[mask_title]
    if dur_col:
        mask_dur = df[dur_col].fillna(1e9) >= 70  # keep >=70s
        df = df[mask_dur]
    print(f"Dropped likely Shorts: {before - len(df)} rows (now {len(df)})")

# Targets: log2(view+1) -> [0,1]
df["log2_views"] = np.log2(df["viewCount"] + 1.0)
view_min = float(df["log2_views"].min())
view_max = float(df["log2_views"].max())
scale = (view_max - view_min) if view_max > view_min else 1.0
df["y_norm"] = (df["log2_views"] - view_min) / scale

print(f"Rows after filter: {len(df)}")
print(f"log2 range: [{view_min:.2f}, {view_max:.2f}]")

# ----------------------------
# Split (Train / Val / Test)
# ----------------------------
train_df, temp_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ----------------------------
# Generators
# ----------------------------
img_size = (args.img_size, args.img_size)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def make_gen(df_part, shuffle, name):
    gen = (train_datagen if shuffle else eval_datagen).flow_from_dataframe(
        df_part,
        directory=args.thumb_dir,
        x_col="thumbnail_file",
        y_col="y_norm",
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="raw",
        shuffle=shuffle
    )
    print(f"{name}: steps/epoch ~ {int(np.ceil(len(df_part) / args.batch_size))}")
    return gen

train_gen = make_gen(train_df, shuffle=True,  name="Train")
val_gen   = make_gen(val_df,   shuffle=False, name="Val")
test_gen  = make_gen(test_df,  shuffle=False, name="Test")

# ----------------------------
# Model
# ----------------------------
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
base.trainable = False  # warm-up

inp = Input(shape=(args.img_size, args.img_size, 3))
x = base(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
x = Dropout(0.30)(x)
x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
x = Dropout(0.20)(x)
x = Dense(1, activation="linear")(x)
out = tf.keras.layers.Activation("linear", dtype="float32", name="fp32_out")(x)  # keep fp32 under mixed precision
model = Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary(print_fn=lambda s: print(s))

# ----------------------------
# Callback: evaluate TEST set every epoch
# ----------------------------
class TestEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_gen, test_df, view_min, scale):
        super().__init__()
        self.test_gen = test_gen
        self.test_df = test_df.reset_index(drop=True)
        self.view_min = view_min
        self.scale = scale
        self.history = {"test_loss": [], "test_mae": [], "test_r2_views": []}

    @staticmethod
    def _r2(y, yhat):
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    def on_epoch_end(self, epoch, logs=None):
        # loss/mae on normalized y
        loss, mae = self.model.evaluate(self.test_gen, verbose=0)
        self.history["test_loss"].append(float(loss))
        self.history["test_mae"].append(float(mae))

        # R² in raw views (denorm + exp2)
        y_pred_norm = self.model.predict(self.test_gen, verbose=0).flatten()
        y_true_norm = self.test_df["y_norm"].values[:len(y_pred_norm)]
        y_pred_log2 = y_pred_norm * self.scale + self.view_min
        y_true_log2 = y_true_norm * self.scale + self.view_min
        pred_views  = np.maximum(0, (2.0 ** y_pred_log2) - 1.0)
        true_views  = np.maximum(0, (2.0 ** y_true_log2) - 1.0)
        r2 = self._r2(true_views, pred_views)
        self.history["test_r2_views"].append(float(r2))

test_cb = TestEvalCallback(test_gen, test_df, view_min, scale)

# ----------------------------
# Training callbacks
# ----------------------------
best_h5_path     = os.path.join(ckpt_dir, "model_best.h5")
best_keras_path  = os.path.join(ckpt_dir, "model_best.keras")
savedmodel_dir   = os.path.join(ckpt_dir, "saved_model")

cbs = [
    # keep an H5 snapshot (legacy) *and* a modern .keras snapshot
    tf.keras.callbacks.ModelCheckpoint(
        best_h5_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        best_keras_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    test_cb,
]

# ----------------------------
# Train: warm-up (frozen base)
# ----------------------------
hist_warm = model.fit(
    train_gen,
    epochs=args.epochs_warm,
    validation_data=val_gen,
    callbacks=cbs,
    verbose=1
)

# ----------------------------
# Fine-tune: unfreeze last ~80 layers
# ----------------------------
for layer in base.layers[-80:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])

hist_ft = model.fit(
    train_gen,
    epochs=args.epochs_ft,
    validation_data=val_gen,
    callbacks=cbs,
    verbose=1
)

# Merge histories (for plots & save)
history = {}
for k in set(list(hist_warm.history.keys()) + list(hist_ft.history.keys())):
    history[k] = hist_warm.history.get(k, []) + hist_ft.history.get(k, [])
# add per-epoch TEST metrics
history["test_loss"] = test_cb.history["test_loss"]
history["test_mae"]  = test_cb.history["test_mae"]
history["test_r2_views"] = test_cb.history["test_r2_views"]

# ----------------------------
# Save model & normalization
# ----------------------------
# final saves (modern .keras + TF SavedModel export)
try:
    model.save(best_keras_path)              # native .keras
except Exception as e:
    print("Warning: could not save .keras model:", e)
try:
    model.export(savedmodel_dir)             # TF SavedModel for serving/TFLite
except Exception as e:
    print("Warning: could not export SavedModel:", e)

with open(os.path.join(ckpt_dir, "norm.json"), "w") as f:
    json.dump({"view_min": view_min, "view_max": view_max}, f, indent=2)
np.save(os.path.join(ckpt_dir, "training_history.npy"), history, allow_pickle=True)

# ----------------------------
# Final Evaluate & Predict on TEST
# ----------------------------
test_loss, test_mae = model.evaluate(test_gen, verbose=1)
print(f"Test Loss (MSE on 0–1): {test_loss:.4f}")
print(f"Test MAE  (on 0–1):     {test_mae:.4f}")

y_pred_norm = model.predict(test_gen, verbose=1).flatten()
y_true_norm = test_df["y_norm"].values[:len(y_pred_norm)]

# Denorm to log2 and to raw views
y_pred_log2 = y_pred_norm * scale + view_min
y_true_log2 = y_true_norm * scale + view_min
predicted_views = np.maximum(0, (2.0 ** y_pred_log2) - 1.0)
actual_views    = np.maximum(0, (2.0 ** y_true_log2) - 1.0)

# ----------------------------
# Metrics (rich)
# ----------------------------
def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

metrics = {
    "mse_norm": float(np.mean((y_true_norm - y_pred_norm)**2)),
    "mae_norm": float(np.mean(np.abs(y_true_norm - y_pred_norm))),
    "mae_log2": float(np.mean(np.abs(y_true_log2 - y_pred_log2))),
    "pearson_log2": float(pearsonr(y_true_log2, y_pred_log2).statistic),
    "spearman_log2": float(spearmanr(y_true_log2, y_pred_log2).correlation),
    "mse_views": float(np.mean((actual_views - predicted_views)**2)),
    "rmse_views": float(np.sqrt(np.mean((actual_views - predicted_views)**2))),
    "mae_views": float(np.mean(np.abs(actual_views - predicted_views))),
    "r2_views": float(r2_score(actual_views, predicted_views)),
    "log2_min": float(view_min), "log2_max": float(view_max),
    "n_test": int(len(test_df)),
}
with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------
# Save TEST predictions CSV + Top-10 under/over
# ----------------------------
pred_csv = os.path.join(args.outdir, "test_predictions.csv")
pred_df = test_df.reset_index(drop=True).iloc[:len(y_pred_norm)].assign(
    y_true_norm=y_true_norm,
    y_pred_norm=y_pred_norm,
    actual_log2=y_true_log2,
    pred_log2=y_pred_log2,
    actual_views=actual_views,
    predicted_views=predicted_views,
    diff_views=(predicted_views - actual_views)  # positive => overestimate
)
pred_df.to_csv(pred_csv, index=False)

# Top-10 overestimated (largest positive diff) & underestimated (most negative)
top_over = pred_df.sort_values("diff_views", ascending=False).head(10)
top_under = pred_df.sort_values("diff_views", ascending=True).head(10)
top_over.to_csv(os.path.join(args.outdir, "top10_over.csv"), index=False)
top_under.to_csv(os.path.join(args.outdir, "top10_under.csv"), index=False)

# ----------------------------
# Quick random sample table from TEST set
# ----------------------------
rng_sample = test_df.sample(args.n_sample, random_state=7).reset_index(drop=True)
sg = eval_datagen.flow_from_dataframe(
    rng_sample, directory=args.thumb_dir, x_col="thumbnail_file", y_col="y_norm",
    target_size=img_size, batch_size=args.n_sample, shuffle=False, class_mode="raw"
)
s_pred_norm = model.predict(sg, verbose=0).flatten()
s_pred_log2 = s_pred_norm * scale + view_min
s_true_log2 = rng_sample["y_norm"].values * scale + view_min
s_pred_views = np.maximum(0, (2.0 ** s_pred_log2) - 1.0)
s_true_views = np.maximum(0, (2.0 ** s_true_log2) - 1.0)
sample_tbl = pd.DataFrame({
    "thumbnail_file": rng_sample["thumbnail_file"],
    "actual_views": s_true_views.astype(np.int64),
    "predicted_views": s_pred_views.astype(np.int64),
    "diff_views": (s_pred_views - s_true_views).astype(np.int64),
})
sample_tbl.to_csv(os.path.join(args.outdir, "sample20.csv"), index=False)

# ----------------------------
# Thumbnail montages (top under/over)
# ----------------------------
def _fmt_views(n: float) -> str:
    n = float(n)
    if n >= 1e9:  return f"{n/1e9:.1f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{int(n)}"

def _safe_open(path, target=(320, 180)):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", target, (220, 220, 220))   # placeholder if missing
    im = im.resize(target, Image.BILINEAR)
    return im

def save_montage(pred_df_local, thumb_dir, title, out_png, k=12, cols=4,
                 thumb_size=(320, 180), annotate_fontsize=10):
    import math
    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(14, 10 if rows > 2 else 8))
    plt.suptitle(title, fontsize=16)
    for i in range(k):
        ax = plt.subplot(rows, cols, i+1)
        if i < len(pred_df_local):
            r = pred_df_local.iloc[i]
            img_path = os.path.join(thumb_dir, str(r["thumbnail_file"]))
            im = _safe_open(img_path, thumb_size)
            ax.imshow(im)
            a = _fmt_views(r["actual_views"])
            p = _fmt_views(r["predicted_views"])
            ax.set_title(f"A:{a}  |  P:{p}", fontsize=annotate_fontsize, pad=6)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=120)
    plt.close()

under_png = os.path.join(args.outdir, "top_underpredicted.png")
over_png  = os.path.join(args.outdir, "top_overpredicted.png")

under_for_grid = pred_df.assign(gap_log2=(pred_df["actual_log2"] - pred_df["pred_log2"])) \
                        .sort_values("gap_log2", ascending=False).head(12)
over_for_grid  = pred_df.assign(gap_log2=(pred_df["pred_log2"] - pred_df["actual_log2"])) \
                        .sort_values("gap_log2", ascending=False).head(12)

save_montage(under_for_grid, args.thumb_dir,
             title="Most underpredicted (log2 true - pred)",
             out_png=under_png, k=12, cols=4)
save_montage(over_for_grid,  args.thumb_dir,
             title="Most overpredicted (log2 true - pred)",
             out_png=over_png, k=12, cols=4)

# ----------------------------
# PLOTS (EDA + Curves)
# ----------------------------
# 1) Training vs Validation vs TEST curves (Loss & MAE)
curves_png = os.path.join(args.outdir, "training_val_test_curves.png")
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.get("loss", []), label="train")
plt.plot(history.get("val_loss", []), label="val")
plt.plot(history.get("test_loss", []), label="test")
plt.title("Loss (MSE, normalized)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.get("mae", []), label="train")
plt.plot(history.get("val_mae", []), label="val")
plt.plot(history.get("test_mae", []), label="test")
plt.title("MAE (normalized) — lower is better"); plt.xlabel("Epoch"); plt.ylabel("MAE")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(curves_png, dpi=140)

# Optional: TEST R² per epoch (to see generalization trend)
r2_curve_png = os.path.join(args.outdir, "test_r2_per_epoch.png")
if len(history.get("test_r2_views", [])) > 0:
    plt.figure(figsize=(8,5))
    plt.plot(history["test_r2_views"], marker="o")
    plt.title("TEST R² per epoch (views)")
    plt.xlabel("Epoch"); plt.ylabel("R² (higher is better)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(r2_curve_png, dpi=140)

# 2) Pred vs Actual (log2)
pva_log2_png = os.path.join(args.outdir, "pred_vs_actual_log2.png")
plt.figure(figsize=(10,7))
plt.scatter(y_true_log2, y_pred_log2, s=8, alpha=0.45)
mn, mx = min(y_true_log2.min(), y_pred_log2.min()), max(y_true_log2.max(), y_pred_log2.max())
plt.plot([mn, mx], [mn, mx], "r--", lw=2)
plt.title("Pred vs Actual (log2 views)")
plt.xlabel("Actual log2(views)"); plt.ylabel("Predicted log2(views)")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(pva_log2_png, dpi=140)

# 3) Pred vs Actual (raw views, log–log)
pva_loglog_png = os.path.join(args.outdir, "pred_vs_actual_loglog.png")
plt.figure(figsize=(10,7))
plt.scatter(actual_views, predicted_views, s=8, alpha=0.4)
plt.xscale("log"); plt.yscale("log")
mx = max(actual_views.max(), predicted_views.max())
plt.plot([1, mx], [1, mx], "r--", lw=2)
plt.title("Pred vs Actual (raw views, log–log)")
plt.xlabel("Actual views (log)"); plt.ylabel("Predicted views (log)")
plt.grid(True, which="both", alpha=.3); plt.tight_layout(); plt.savefig(pva_loglog_png, dpi=140)

# 4) Error distribution (views)
err_hist_png = os.path.join(args.outdir, "error_hist_views.png")
plt.figure(figsize=(10,6))
errors = predicted_views - actual_views
clip = 10_000_000
plt.hist(np.clip(errors, -clip, clip), bins=80)
plt.title("Error Distribution (views)")
plt.xlabel("Pred - Actual (views) [clipped ±10M]"); plt.ylabel("Freq")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(err_hist_png, dpi=140)

# 5) Residuals vs Prediction (log2)
resid_png = os.path.join(args.outdir, "residuals_log2.png")
plt.figure(figsize=(10,6))
resid = (y_true_log2 - y_pred_log2)
plt.scatter(y_pred_log2, resid, s=6, alpha=0.35)
plt.axhline(0, color="r", ls="--")
plt.title("Residuals vs Prediction (log2)")
plt.xlabel("Predicted log2(views)"); plt.ylabel("Residual (true - pred)")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(resid_png, dpi=140)

# 6) MAE by popularity decile (log2)
bucket_png = os.path.join(args.outdir, "bucket_mae_log2.png")
q = np.quantile(y_true_log2, np.linspace(0,1,11))
bins = pd.cut(y_true_log2, q, include_lowest=True, labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)])
bucket_mae = pd.Series(np.abs(resid)).groupby(bins).mean()
plt.figure(figsize=(12,6))
bucket_mae.plot(kind="bar")
plt.title("MAE by popularity decile (log2)")
plt.ylabel("MAE (log2)"); plt.tight_layout(); plt.savefig(bucket_png, dpi=140)

print(f"\nSaved:")
print(f"- Checkpoints (.h5):                 {best_h5_path}")
print(f"- Best model (.keras):               {best_keras_path}")
print(f"- SavedModel export dir:             {savedmodel_dir}")
print(f"- Norm params:                       {os.path.join(ckpt_dir, 'norm.json')}")
print(f"- Training history (.npy):           {os.path.join(ckpt_dir, 'training_history.npy')}")
print(f"- Metrics:                           {os.path.join(args.outdir, 'metrics.json')}")
print(f"- Curves (train/val/test):           {curves_png}")
print(f"- TEST R² per epoch:                 {r2_curve_png if len(history.get('test_r2_views', [])) else 'n/a'}")
print(f"- Pred vs Actual (log2):             {pva_log2_png}")
print(f"- Pred vs Actual (log–log views):    {pva_loglog_png}")
print(f"- Error histogram:                   {err_hist_png}")
print(f"- Residuals plot:                    {resid_png}")
print(f"- MAE by decile:                     {bucket_png}")
print(f"- TEST predictions CSV:              {pred_csv}")
print(f"- Top10 overestimated:               {os.path.join(args.outdir, 'top10_over.csv')}")
print(f"- Top10 underestimated:              {os.path.join(args.outdir, 'top10_under.csv')}")
print(f"- Underpredicted montage:            {under_png}")
print(f"- Overpredicted montage:             {over_png}")
print(f"- Sample {args.n_sample} CSV:        {os.path.join(args.outdir, 'sample20.csv')}")