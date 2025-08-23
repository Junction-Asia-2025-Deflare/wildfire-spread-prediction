from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model

# ----------------------------
# Constants
# ----------------------------
INPUT_SHAPE = (64, 64, 12)
OUTPUT_SHAPE = (64, 64, 1)
POST_THRESHOLD = 0.8

# ----------------------------
# Multi-kernel encoder block
# ----------------------------

def multi_kernel_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Applies Conv(3×3), Conv(5×5), Conv(7×7) → concat → MaxPool(2×2).
    All conv layers: stride=1, padding='same', activation=ReLU.
    """
    c3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', name=f"{name}_conv3")(x)
    c5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu', name=f"{name}_conv5")(x)
    c7 = layers.Conv2D(filters, (7, 7), padding='same', activation='relu', name=f"{name}_conv7")(x)
    concat = layers.Concatenate(name=f"{name}_concat")([c3, c5, c7])
    pooled = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(concat)
    return pooled

# ----------------------------
# Model builder
# ----------------------------

def build_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> Model:
    inp = layers.Input(shape=input_shape, name="input_tile")

    # Encoder
    x1 = multi_kernel_block(inp, 16, name="B1")
    x2 = multi_kernel_block(x1, 32, name="B2")
    x3 = multi_kernel_block(x2, 64, name="B3")
    x4 = multi_kernel_block(x3, 128, name="B4")
    x5 = multi_kernel_block(x4, 256, name="B5")

    # Decoder (업샘플링 경로)
    d4 = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x5)
    d4 = layers.Concatenate()([d4, x4])   # skip connection
    d3 = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(d4)
    d3 = layers.Concatenate()([d3, x3])
    d2 = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(d3)
    d2 = layers.Concatenate()([d2, x2])
    d1 = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(d2)
    d1 = layers.Concatenate()([d1, x1])
    d1_up = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(d1)

    out = layers.Conv2D(1, (1,1), activation="sigmoid")(d1_up)

    model = Model(inputs=inp, outputs=out, name="MultiKernelCNN_UNet")
    return model

# ----------------------------
# Generalized Dice Loss with mask
# ----------------------------

def generalized_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask out unlabeled pixels (-1)
    valid_mask = tf.where(y_true >= 0.0, 1.0, 0.0)
    y_true_valid = y_true * valid_mask
    y_pred_valid = y_pred * valid_mask

    # Flatten over spatial dims
    y_true_f = tf.reshape(y_true_valid, [tf.shape(y_true_valid)[0], -1])
    y_pred_f = tf.reshape(y_pred_valid, [tf.shape(y_pred_valid)[0], -1])

    # Compute class volumes across batch for binary classes 0 and 1
    # For GDL, we need per-class weights. We'll compute for foreground (1) and background (0).
    eps = 1e-7
    # Foreground
    y_true_fg = y_true_f
    y_pred_fg = y_pred_f
    # Background (1 - y)
    y_true_bg = 1.0 - y_true_f
    y_pred_bg = 1.0 - y_pred_f

    vol_fg = tf.reduce_sum(y_true_fg, axis=1) + eps
    vol_bg = tf.reduce_sum(y_true_bg, axis=1) + eps

    w_fg = 1.0 / tf.square(vol_fg)
    w_bg = 1.0 / tf.square(vol_bg)

    # Intersection and union terms
    intersect_fg = tf.reduce_sum(y_true_fg * y_pred_fg, axis=1)
    intersect_bg = tf.reduce_sum(y_true_bg * y_pred_bg, axis=1)

    denom_fg = tf.reduce_sum(y_true_fg + y_pred_fg, axis=1) + eps
    denom_bg = tf.reduce_sum(y_true_bg + y_pred_bg, axis=1) + eps

    numerator = 2.0 * (w_fg * intersect_fg + w_bg * intersect_bg)
    denominator = (w_fg * denom_fg + w_bg * denom_bg) + eps

    gdl = 1.0 - tf.reduce_mean(numerator / denominator)
    return gdl

# ----------------------------
# Metrics with masking (thresholded at 0.8 per paper)
# ----------------------------

def _apply_mask_threshold(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    valid = tf.where(y_true >= 0.0, 1.0, 0.0)  # -1인 픽셀 제외
    y_pred_bin = tf.where(y_pred >= threshold, 1.0, 0.0)
    return y_true, y_pred_bin, valid

@tf.function
def masked_precision(y_true, y_pred, threshold=POST_THRESHOLD):
    y_true, y_pred_bin, valid = _apply_mask_threshold(y_true, y_pred, threshold)
    tp = tf.reduce_sum(y_pred_bin * y_true * valid)
    fp = tf.reduce_sum(y_pred_bin * (1.0 - y_true) * valid)
    return tf.math.divide_no_nan(tp, tp + fp)

@tf.function
def masked_recall(y_true, y_pred, threshold=POST_THRESHOLD):
    y_true, y_pred_bin, valid = _apply_mask_threshold(y_true, y_pred, threshold)
    tp = tf.reduce_sum(y_pred_bin * y_true * valid)
    fn = tf.reduce_sum((1.0 - y_pred_bin) * y_true * valid)
    return tf.math.divide_no_nan(tp, tp + fn)

@tf.function
def masked_f1(y_true, y_pred, threshold=POST_THRESHOLD):
    p = masked_precision(y_true, y_pred, threshold)
    r = masked_recall(y_true, y_pred, threshold)
    return tf.math.divide_no_nan(2.0 * p * r, p + r)

@tf.function
def masked_overall_accuracy(y_true, y_pred, threshold=POST_THRESHOLD):
    y_true, y_pred_bin, valid = _apply_mask_threshold(y_true, y_pred, threshold)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred_bin), tf.float32) * valid)
    total = tf.reduce_sum(valid)
    return tf.math.divide_no_nan(correct, total)

# ----------------------------
# Compile helper
# ----------------------------

def compile_model(model: Model, lr: float = 6.25e-5) -> Model:
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=generalized_dice_loss,
        metrics=[masked_precision, masked_recall, masked_f1, masked_overall_accuracy]
    )
    return model

# ----------------------------
# Training utility (example)
# ----------------------------

def train(model: Model,
          train_ds: tf.data.Dataset,
          val_ds: tf.data.Dataset,
          epochs: int = 250,
          steps_per_epoch: int | None = None,
          validation_steps: int | None = None,
          callbacks: list | None = None):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    return history

# ----------------------------
# Inference + Post-processing
# ----------------------------

def predict_binarized(model: Model, x: tf.Tensor, threshold: float = POST_THRESHOLD) -> tf.Tensor:
    prob = model.predict(x)
    bin_mask = (prob >= threshold).astype('float32')
    return bin_mask

# ----------------------------
# If run as script, build & summarize
# ----------------------------
if __name__ == "__main__":
    model = build_model()
    model = compile_model(model)
    model.summary()