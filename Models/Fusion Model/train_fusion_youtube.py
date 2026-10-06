#!/usr/bin/env python3
"""
Fusion training script:

- CNN thumbnail regression (transfer learning)
- Tabular regression (GradientBoosting) on video/channel metadata
- Fusion model (LinearRegression) on top of CNN + Tabular predictions

Modes:
- --mode train : train all models, save them, compute metrics/plots
- --mode infer : load saved models, recompute predictions/metrics/plots only

Outputs (under --outdir):
- cnn_metrics.json
- tabular_metrics.json
- fusion_metrics.json
- fusion_test_predictions.csv
- Plots:
    - cnn_training_curves_loss.png
    - cnn_training_curves_mae.png
    - cnn_scatter_test_log10.png
    - cnn_residuals_hist.png
    - tabular_scatter_test_log10.png
    - tabular_residuals_hist.png
    - fusion_scatter_test_log10.png
    - fusion_residuals_hist.png
- Top-10 over/under-predicted CSVs for each model
"""

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Paths & Defaults
# ----------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

DEFAULTS = {
    "csv":       os.path.normpath(os.path.join(PROJECT_ROOT, "Dataset/Mohamed_Dataset/youtube_dataset_filtered.csv")),
    "thumb_dir": os.path.normpath(os.path.join(PROJECT_ROOT, "Dataset/Mohamed_Dataset/thumbnails")),
    "outdir":    os.path.normpath(os.path.join(PROJECT_ROOT, "outputs_fusion")),
    "img_size":  224,
    "batch_size": 16,
    "epochs_warm": 5,
    "epochs_ft": 15,
    "val_split": 0.30,
    "seed": 42,
}

parser = argparse.ArgumentParser(description="Fusion: Thumbnail CNN + Tabular + Stacking")
parser.add_argument("--csv",        default=DEFAULTS["csv"], help="Path to dataset CSV")
parser.add_argument("--thumb_dir",  default=DEFAULTS["thumb_dir"], help="Directory containing thumbnail images")
parser.add_argument("--outdir",     default=DEFAULTS["outdir"], help="Output directory")
parser.add_argument("--img_size",   type=int, default=DEFAULTS["img_size"], help="Image size")
parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"], help="Batch size")
parser.add_argument("--epochs_warm",type=int, default=DEFAULTS["epochs_warm"], help="Warm-up epochs (CNN frozen)")
parser.add_argument("--epochs_ft",  type=int, default=DEFAULTS["epochs_ft"], help="Fine-tune epochs")
parser.add_argument("--val_split",  type=float, default=DEFAULTS["val_split"], help="Val+Test fraction")
parser.add_argument("--seed",       type=int, default=DEFAULTS["seed"], help="Random seed")
parser.add_argument(
    "--backbone",
    choices=["b0", "b2", "resnet", "convnext"],
    default="convnext",
    help="CNN backbone"
)
parser.add_argument("--drop_shorts", action="store_true",
                    help="Drop shorts (#shorts/#fyp or duration_sec<70 if columns exist)")
parser.add_argument(
    "--mode",
    choices=["train", "infer"],
    default="train",
    help="train = train all models, infer = load saved models and skip training"
)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
ckpt_dir = os.path.join(args.outdir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# ----------------------------
# TensorFlow / ML imports
# ----------------------------
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB2,
    ResNet50,
    ConvNeXtTiny,
)
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr, spearmanr

# ----------------------------
# Reproducibility & TF setup
# ----------------------------
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)

mixed_precision.set_global_policy("mixed_float16")

# You can comment this out if you want to avoid XLA logs
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print("TF:", tf.__version__)
print("Mode:", args.mode)
print("CSV:", args.csv)
print("Thumb dir:", args.thumb_dir)
print("Out dir:", args.outdir)
print("Backbone:", args.backbone)

# ----------------------------
# Load & clean data (shared for CNN + Tabular)
# ----------------------------
df = pd.read_csv(args.csv, low_memory=False)

required = ["thumbnail_file", "viewCount"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in CSV (columns: {df.columns.tolist()})")

df = df[df["thumbnail_file"].notna() & (df["thumbnail_file"] != "#NAME?") & df["viewCount"].notna()]
df["viewCount"] = df["viewCount"].astype(float)

# Title as string
if "title" in df.columns:
    df["title"] = df["title"].astype(str)
else:
    df["title"] = ""

# Date features (for tabular)
if "publishedAt" in df.columns:
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["pub_year"] = df["publishedAt"].dt.year
    df["pub_month"] = df["publishedAt"].dt.month
    df["pub_day"] = df["publishedAt"].dt.day
    df["pub_weekday"] = df["publishedAt"].dt.weekday
    df["pub_hour"] = df["publishedAt"].dt.hour
    df["days_since_first_video"] = (df["publishedAt"] - df["publishedAt"].min()).dt.days
else:
    df["pub_year"] = 0
    df["pub_month"] = 0
    df["pub_day"] = 0
    df["pub_weekday"] = 0
    df["pub_hour"] = 0
    df["days_since_first_video"] = 0

# Duration to seconds (for tabular)
def duration_to_seconds(d):
    import re
    if not isinstance(d, str):
        return 0
    h = int(re.search(r"(\d+)H", d).group(1)) if "H" in d else 0
    m = int(re.search(r"(\d+)M", d).group(1)) if "M" in d else 0
    s = int(re.search(r"(\d+)S", d).group(1)) if "S" in d else 0
    return h * 3600 + m * 60 + s

if "duration" in df.columns:
    df["duration_sec"] = df["duration"].apply(duration_to_seconds)
else:
    df["duration_sec"] = 0

# Shorts filter (optional)
if args.drop_shorts:
    before = len(df)
    if "title" in df.columns:
        mask_title = ~df["title"].fillna("").str.lower().str.contains(r"#shorts|#fyp")
        df = df[mask_title]
    if "duration_sec" in df.columns:
        df = df[df["duration_sec"].fillna(1e9) >= 70]
    print(f"Dropped likely Shorts: {before - len(df)} rows (now {len(df)})")

# Targets
df["log2_views"] = np.log2(df["viewCount"] + 1.0)      # for CNN
view_min = float(df["log2_views"].min())
view_max = float(df["log2_views"].max())
scale = (view_max - view_min) if view_max > view_min else 1.0
df["y_norm"] = (df["log2_views"] - view_min) / scale

df["log_views"] = np.log1p(df["viewCount"])            # for tabular (ln(view+1))

print(f"Rows after filter: {len(df)}")
print(f"log2 range: [{view_min:.2f}, {view_max:.2f}]")

# ----------------------------
# Split once: Train / Val / Test (shared)
# ----------------------------
train_df, temp_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ----------------------------
# CNN: image generators
# ----------------------------
img_size = (args.img_size, args.img_size)

if args.backbone in ["b0", "b2"]:
    preprocess_fn = effnet_preprocess
elif args.backbone == "resnet":
    preprocess_fn = resnet_preprocess
else:
    preprocess_fn = convnext_preprocess

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

def make_gen(df_part, shuffle, name):
    gen = (train_datagen if shuffle else eval_datagen).flow_from_dataframe(
        df_part,
        directory=args.thumb_dir,
        x_col="thumbnail_file",
        y_col="y_norm",
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="raw",
        shuffle=shuffle,
    )
    print(f"{name}: {len(df_part)} rows, steps/epoch ~ {int(np.ceil(len(df_part)/args.batch_size))}")
    return gen

train_gen = make_gen(train_df, shuffle=True,  name="Train")
val_gen   = make_gen(val_df,   shuffle=False, name="Val")
test_gen  = make_gen(test_df,  shuffle=False, name="Test")

# ----------------------------
# CNN model: choose backbone
# ----------------------------
if args.backbone == "b2":
    if args.img_size < 256:
        print(f"[Note] EfficientNetB2 works best at 260×260. You passed {args.img_size}.")
    base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
    ft_unfreeze = 120
elif args.backbone == "b0":
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
    ft_unfreeze = 80
elif args.backbone == "resnet":
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
    ft_unfreeze = 60
elif args.backbone == "convnext":
    base = ConvNeXtTiny(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
    ft_unfreeze = 80
else:
    raise ValueError(f"Unsupported backbone: {args.backbone}")

base.trainable = False

inp = Input(shape=(args.img_size, args.img_size, 3))
x = base(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
x = Dropout(0.30)(x)
x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
x = Dropout(0.20)(x)
x = Dense(1, activation="linear")(x)
out = tf.keras.layers.Activation("linear", dtype="float32", name="fp32_out")(x)
cnn_model = Model(inp, out)

loss_fn = tf.keras.losses.MeanSquaredError()

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["mae"],
)

cnn_model.summary()

best_cnn_path = os.path.join(ckpt_dir, "cnn_model_best.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_cnn_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

hist_warm, hist_ft = None, None

# ----------------------------
# CNN training / loading
# ----------------------------
if args.mode == "train":
    print("\n=== Training CNN (warm-up) ===")
    hist_warm = cnn_model.fit(
        train_gen,
        epochs=args.epochs_warm,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n=== Fine-tuning CNN ===")
    for layer in base.layers[-ft_unfreeze:]:
        layer.trainable = True

    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=["mae"],
    )

    hist_ft = cnn_model.fit(
        train_gen,
        epochs=args.epochs_ft,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
else:
    print("\n=== Loading saved CNN weights ===")
    if not os.path.exists(best_cnn_path):
        raise FileNotFoundError(f"Saved CNN model not found: {best_cnn_path}")
    cnn_model.load_weights(best_cnn_path)

# ----------------------------
# CNN evaluation on VAL & TEST (in raw views)
# ----------------------------
def denorm_to_views(y_norm, view_min, scale):
    y_log2 = y_norm * scale + view_min
    return np.maximum(0, (2.0 ** y_log2) - 1.0), y_log2

# VAL
y_val_norm_true = val_df["y_norm"].values
y_val_norm_pred = cnn_model.predict(val_gen, verbose=1).flatten()
val_views_true, val_log2_true = denorm_to_views(y_val_norm_true, view_min, scale)
val_views_pred_cnn, val_log2_pred = denorm_to_views(y_val_norm_pred, view_min, scale)

# TEST
y_test_norm_true = test_df["y_norm"].values
y_test_norm_pred = cnn_model.predict(test_gen, verbose=1).flatten()
test_views_true, test_log2_true = denorm_to_views(y_test_norm_true, view_min, scale)
test_views_pred_cnn, test_log2_pred = denorm_to_views(y_test_norm_pred, view_min, scale)

def r2_numpy(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

cnn_metrics = {
    "test_mse_views": float(mean_squared_error(test_views_true, test_views_pred_cnn)),
    "test_rmse_views": float(np.sqrt(mean_squared_error(test_views_true, test_views_pred_cnn))),
    "test_mae_views": float(mean_absolute_error(test_views_true, test_views_pred_cnn)),
    "test_r2_views": float(r2_numpy(test_views_true, test_views_pred_cnn)),
    "test_pearson_log2": float(pearsonr(test_log2_true, test_log2_pred).statistic),
    "test_spearman_log2": float(spearmanr(test_log2_true, test_log2_pred).correlation),
}

with open(os.path.join(args.outdir, "cnn_metrics.json"), "w") as f:
    json.dump(cnn_metrics, f, indent=2)

print("\n=== CNN TEST METRICS ===")
for k, v in cnn_metrics.items():
    print(f"{k}: {v:.4f}")

# ----------------------------
# TABULAR MODEL (Gradient Boosting on metadata)
# ----------------------------
print("\n=== Tabular Model (GradientBoostingRegressor) ===")

# Work on copies so we don’t mess up CNN dfs
train_tab = train_df.copy()
val_tab   = val_df.copy()
test_tab  = test_df.copy()

# Title length
train_tab["title"] = train_tab["title"].astype(str)
val_tab["title"]   = val_tab["title"].astype(str)
test_tab["title"]  = test_tab["title"].astype(str)

train_tab["title_length"] = train_tab["title"].str.len().fillna(0)
val_tab["title_length"]   = val_tab["title"].str.len().fillna(0)
test_tab["title_length"]  = test_tab["title"].str.len().fillna(0)

# Channel stats (only if channel_id exists)
if "channel_id" in train_tab.columns:
    ch_stats = train_tab.groupby("channel_id")["viewCount"].agg(["mean", "median", "std", "count"]).fillna(0)
    ch_stats.columns = [f"channel_{c}_views" for c in ch_stats.columns]

    train_tab = train_tab.merge(ch_stats, on="channel_id", how="left")
    val_tab   = val_tab.merge(ch_stats, on="channel_id", how="left")
    test_tab  = test_tab.merge(ch_stats, on="channel_id", how="left")

    for col in ch_stats.columns:
        val_tab[col]  = val_tab[col].fillna(0)
        test_tab[col] = test_tab[col].fillna(0)
else:
    for c in ["mean", "median", "std", "count"]:
        col_name = f"channel_{c}_views"
        train_tab[col_name] = 0
        val_tab[col_name]   = 0
        test_tab[col_name]  = 0

numeric_features = [
    "duration_sec",
    "pub_year",
    "pub_month",
    "pub_day",
    "pub_weekday",
    "pub_hour",
    "days_since_first_video",
    "likeCount",
    "commentCount",
    "channel_mean_views",
    "channel_median_views",
    "channel_std_views",
    "channel_count_views",
    "title_length",
]

# Ensure numeric features exist
for df_tmp in [train_tab, val_tab, test_tab]:
    for col in numeric_features:
        if col not in df_tmp.columns:
            df_tmp[col] = 0.0

cat_cols = [c for c in ["definition", "caption", "category"] if c in df.columns]

# One-hot encoding
train_tab = pd.get_dummies(train_tab, columns=cat_cols, drop_first=True)
val_tab   = pd.get_dummies(val_tab,   columns=cat_cols, drop_first=True)
test_tab  = pd.get_dummies(test_tab,  columns=cat_cols, drop_first=True)

train_cols = train_tab.columns
val_tab  = val_tab.reindex(columns=train_cols, fill_value=0)
test_tab = test_tab.reindex(columns=train_cols, fill_value=0)

encoded_cat_cols = [c for c in train_tab.columns if any(col in c for col in cat_cols)]
feature_cols = [c for c in numeric_features if c in train_tab.columns] + encoded_cat_cols

X_train_tab = train_tab[feature_cols].astype(float).fillna(0)
X_val_tab   = val_tab[feature_cols].astype(float).fillna(0)
X_test_tab  = test_tab[feature_cols].astype(float).fillna(0)

# Scale ONLY numeric features
scaler = StandardScaler()
numeric_present = [c for c in numeric_features if c in X_train_tab.columns]

X_train_tab[numeric_present] = scaler.fit_transform(X_train_tab[numeric_present])
X_val_tab[numeric_present]   = scaler.transform(X_val_tab[numeric_present])
X_test_tab[numeric_present]  = scaler.transform(X_test_tab[numeric_present])

# Target in log-space
y_train_tab = train_tab["log_views"].values   # ln(views+1)
y_val_tab   = val_tab["log_views"].values
y_test_tab  = test_tab["log_views"].values

tab_model_path = os.path.join(args.outdir, "tabular_model.pkl")

if args.mode == "train":
    print("\n=== Training Tabular Model ===")
    tab_model = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    tab_model.fit(X_train_tab, y_train_tab)
    with open(tab_model_path, "wb") as f:
        pickle.dump(tab_model, f)
else:
    print("\n=== Loading saved Tabular Model ===")
    if not os.path.exists(tab_model_path):
        raise FileNotFoundError(f"Saved tabular model not found: {tab_model_path}")
    with open(tab_model_path, "rb") as f:
        tab_model = pickle.load(f)

# Predictions (log space -> views)
val_log_pred_tab  = tab_model.predict(X_val_tab)
test_log_pred_tab = tab_model.predict(X_test_tab)

val_views_pred_tab  = np.expm1(val_log_pred_tab)
test_views_pred_tab = np.expm1(test_log_pred_tab)

val_views_true_tab  = np.expm1(y_val_tab)
test_views_true_tab = np.expm1(y_test_tab)

tab_metrics = {
    "val_r2_log": float(r2_score(y_val_tab, val_log_pred_tab)),
    "val_mae_log": float(mean_absolute_error(y_val_tab, val_log_pred_tab)),
    "test_mse_views": float(mean_squared_error(test_views_true_tab, test_views_pred_tab)),
    "test_rmse_views": float(np.sqrt(mean_squared_error(test_views_true_tab, test_views_pred_tab))),
    "test_mae_views": float(mean_absolute_error(test_views_true_tab, test_views_pred_tab)),
    "test_r2_views": float(r2_score(test_views_true_tab, test_views_pred_tab)),
}

with open(os.path.join(args.outdir, "tabular_metrics.json"), "w") as f:
    json.dump(tab_metrics, f, indent=2)

print("\n=== TABULAR TEST METRICS (GradientBoosting) ===")
for k, v in tab_metrics.items():
    print(f"{k}: {v:.4f}")

# ----------------------------
# FUSION MODEL (LinearRegression on CNN + Tabular predictions)
# ----------------------------
print("\n=== Fusion Model (LinearRegression on CNN + Tabular predictions) ===")

# Use VAL set to train fusion (views space)
X_fusion_val = np.vstack([
    val_views_pred_cnn,   # CNN prediction on VAL
    val_views_pred_tab,   # Tabular prediction on VAL
]).T
y_fusion_val = val_views_true   # from CNN pipeline (true views on VAL)

fusion_model_path = os.path.join(args.outdir, "fusion_model.pkl")

if args.mode == "train":
    print("\n=== Training Fusion Model ===")
    fusion_model = LinearRegression()
    fusion_model.fit(X_fusion_val, y_fusion_val)
    with open(fusion_model_path, "wb") as f:
        pickle.dump(fusion_model, f)
else:
    print("\n=== Loading saved Fusion Model ===")
    if not os.path.exists(fusion_model_path):
        raise FileNotFoundError(f"Saved fusion model not found: {fusion_model_path}")
    with open(fusion_model_path, "rb") as f:
        fusion_model = pickle.load(f)

# Evaluate on TEST
X_fusion_test = np.vstack([
    test_views_pred_cnn,
    test_views_pred_tab,
]).T
fusion_pred_views_test = fusion_model.predict(X_fusion_test)

fusion_metrics = {
    "test_mse_views": float(mean_squared_error(test_views_true, fusion_pred_views_test)),
    "test_rmse_views": float(np.sqrt(mean_squared_error(test_views_true, fusion_pred_views_test))),
    "test_mae_views": float(mean_absolute_error(test_views_true, fusion_pred_views_test)),
    "test_r2_views": float(r2_score(test_views_true, fusion_pred_views_test)),
}

with open(os.path.join(args.outdir, "fusion_metrics.json"), "w") as f:
    json.dump(fusion_metrics, f, indent=2)

print("\n=== FUSION TEST METRICS ===")
for k, v in fusion_metrics.items():
    print(f"{k}: {v:.4f}")

# ----------------------------
# Save combined TEST predictions CSV
# ----------------------------
fusion_pred_df = test_df.reset_index(drop=True).copy()
fusion_pred_df = fusion_pred_df.assign(
    actual_views=test_views_true,
    cnn_pred_views=test_views_pred_cnn,
    tabular_pred_views=test_views_pred_tab,
    fusion_pred_views=fusion_pred_views_test,
)

fusion_csv_path = os.path.join(args.outdir, "fusion_test_predictions.csv")
fusion_pred_df.to_csv(fusion_csv_path, index=False)

print("\nSaved base CSV:")
print(f"- Fusion predictions:   {fusion_csv_path}")

# ----------------------------
# PLOTS & TOP OVER/UNDER-PREDICTED
# ----------------------------

# 1) CNN training curves
def plot_training_curves(hist_warm, hist_ft, outdir):
    if hist_warm is None or hist_ft is None:
        print("Skipping CNN training curves (no history, probably running in infer mode).")
        return

    train_loss = hist_warm.history["loss"] + hist_ft.history["loss"]
    val_loss   = hist_warm.history["val_loss"] + hist_ft.history["val_loss"]
    train_mae  = hist_warm.history["mae"] + hist_ft.history["mae"]
    val_mae    = hist_warm.history["val_mae"] + hist_ft.history["val_mae"]

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cnn_training_curves_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_mae, label="Train MAE")
    plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("CNN Training vs Validation MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cnn_training_curves_mae.png"))
    plt.close()

plot_training_curves(hist_warm, hist_ft, args.outdir)

# Helper: log10-safe transform
def safe_log10(x):
    return np.log10(np.maximum(x, 1.0))

# 2) Scatter + residual plots for each model
def plot_scatter_and_residuals(true_views, pred_views, name_prefix, outdir):
    # Scatter (log10 space)
    plt.figure(figsize=(6, 6))
    plt.scatter(safe_log10(true_views), safe_log10(pred_views), s=5, alpha=0.3)
    lims = [
        min(safe_log10(true_views).min(), safe_log10(pred_views).min()),
        max(safe_log10(true_views).max(), safe_log10(pred_views).max()),
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True log10(views)")
    plt.ylabel("Predicted log10(views)")
    plt.title(f"{name_prefix}: True vs Predicted (log10 views)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name_prefix.lower()}_scatter_test_log10.png"))
    plt.close()

    # Residuals histogram (views space)
    residuals = pred_views - true_views
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=100, alpha=0.8)
    plt.xlabel("Prediction Error (pred - true) [views]")
    plt.ylabel("Count")
    plt.title(f"{name_prefix}: Residuals (views)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name_prefix.lower()}_residuals_hist.png"))
    plt.close()

# CNN plots
plot_scatter_and_residuals(test_views_true, test_views_pred_cnn, "CNN", args.outdir)
# Tabular plots
plot_scatter_and_residuals(test_views_true_tab, test_views_pred_tab, "TABULAR", args.outdir)
# Fusion plots
plot_scatter_and_residuals(test_views_true, fusion_pred_views_test, "FUSION", args.outdir)

# 3) Top over/under-predicted for each model
def save_top_over_under(df_base, true_col, pred_col, name_prefix, outdir, k=10):
    errors = df_base[pred_col] - df_base[true_col]
    df_err = df_base.copy()
    df_err["error"] = errors
    df_err["abs_error"] = np.abs(errors)

    over = df_err.sort_values("error", ascending=False).head(k)
    under = df_err.sort_values("error", ascending=True).head(k)

    # Keep some useful columns
    cols_to_keep = [c for c in ["video_id", "title", "thumbnail_file",
                                true_col, pred_col, "error", "abs_error"]
                    if c in df_err.columns]

    over[cols_to_keep].to_csv(
        os.path.join(outdir, f"top_{k}_overpredicted_{name_prefix.lower()}.csv"),
        index=False,
    )
    under[cols_to_keep].to_csv(
        os.path.join(outdir, f"top_{k}_underpredicted_{name_prefix.lower()}.csv"),
        index=False,
    )

# Use fusion_pred_df (has actual + all preds)
save_top_over_under(fusion_pred_df, "actual_views", "cnn_pred_views", "CNN", args.outdir, k=10)
save_top_over_under(fusion_pred_df, "actual_views", "tabular_pred_views", "TABULAR", args.outdir, k=10)
save_top_over_under(fusion_pred_df, "actual_views", "fusion_pred_views", "FUSION", args.outdir, k=10)

print("\nSaved plots & analysis:")
print(f"- CNN training curves (loss/MAE) [train mode only]")
print(f"- Scatter + residual plots for CNN, Tabular, Fusion")
print(f"- Top over/under-predicted CSVs for CNN, Tabular, Fusion")
print(f"Output directory: {args.outdir}")
