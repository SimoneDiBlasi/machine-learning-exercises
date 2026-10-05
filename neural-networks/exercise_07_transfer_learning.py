from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# Exercise 07 - Transfer Learning for Image Classification
# ============================================================

# ------------------------------------------------------------
# Step 1 - Configuration
# ------------------------------------------------------------

IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


# ------------------------------------------------------------
# Step 2 - Download Dataset
# ------------------------------------------------------------

dataset_path = kagglehub.dataset_download(
    "shaunthesheep/microsoft-catsvsdogs-dataset"
)

dataset_path = Path(dataset_path)

print("Dataset path:")
print(dataset_path)


# ------------------------------------------------------------
# Step 3 - Find Cat and Dog Directories
# ------------------------------------------------------------

cat_dirs = list(dataset_path.rglob("Cat"))
dog_dirs = list(dataset_path.rglob("Dog"))

if not cat_dirs or not dog_dirs:
    raise FileNotFoundError(
        "Could not find Cat and Dog directories."
    )

data_dir = cat_dirs[0].parent

print("Data directory:")
print(data_dir)


# ------------------------------------------------------------
# Step 4 - Load Dataset
# ------------------------------------------------------------

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="binary",
    validation_split=VALIDATION_SPLIT,
    subset="both",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ------------------------------------------------------------
# Step 5 - Ignore Corrupted Images
# ------------------------------------------------------------

train_ds = train_ds.apply(
    tf.data.experimental.ignore_errors()
)

val_ds = val_ds.apply(
    tf.data.experimental.ignore_errors()
)


# ------------------------------------------------------------
# Step 6 - Optimize Dataset Loading
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# Step 7 - Load Pre-Trained MobileNetV2
# ------------------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the entire pre-trained network.
base_model.trainable = False


# ------------------------------------------------------------
# Step 8 - Build Transfer Learning Model
# ------------------------------------------------------------

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

inputs = tf.keras.Input(
    shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

x = preprocess_input(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ------------------------------------------------------------
# Step 9 - Compile Model
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Step 10 - Display Model Parameters
# ------------------------------------------------------------

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)

print()
print("Transfer Learning Model")
print("-----------------------")
print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", non_trainable_params)


# ------------------------------------------------------------
# Step 11 - Train Transfer Learning Model
# ------------------------------------------------------------

EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ------------------------------------------------------------
# Step 12 - Evaluate Transfer Learning Model
# ------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print()
print("Transfer Learning Results")
print("-------------------------")
print(f"Validation Loss: {test_loss:.4f}")
print(f"Validation Accuracy: {test_accuracy:.4f}")


# ------------------------------------------------------------
# Step 13 - Plot Training History
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Transfer Learning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Transfer Learning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Step 14 - Confusion Matrix
# ------------------------------------------------------------

y_true = []
y_pred = []
y_probability = []

for images, labels in val_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype("int32")

    y_true.extend(
        labels.numpy()
        .flatten()
        .astype("int32")
    )

    y_pred.extend(
        predictions.flatten()
    )

    y_probability.extend(
        probabilities.flatten()
    )


y_true = tf.convert_to_tensor(y_true)
y_pred = tf.convert_to_tensor(y_pred)
y_probability = tf.convert_to_tensor(y_probability)


confusion_matrix = tf.math.confusion_matrix(
    y_true,
    y_pred
)

print()
print("Confusion Matrix")
print("-----------------")
print(confusion_matrix.numpy())


# ============================================================
# FINE-TUNING
# ============================================================


# ------------------------------------------------------------
# Step 15 - Unfreeze MobileNetV2
# ------------------------------------------------------------

# Allow MobileNetV2 layers to become trainable.
base_model.trainable = True


# Freeze the early layers of MobileNetV2.
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# Keep Batch Normalization layers frozen.
# This helps keep the pre-trained statistics stable
# during fine-tuning.
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


# ------------------------------------------------------------
# Step 16 - Display Fine-Tuning Parameters
# ------------------------------------------------------------

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)

print()
print("Fine-Tuning Configuration")
print("-------------------------")
print("Fine-tune from layer:", fine_tune_at)
print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", non_trainable_params)


# ------------------------------------------------------------
# Step 17 - Recompile Model
# ------------------------------------------------------------

# Use a very small learning rate during fine-tuning.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Step 18 - Fine-Tune Model
# ------------------------------------------------------------

FINE_TUNE_EPOCHS = 5

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS
)


# ------------------------------------------------------------
# Step 19 - Evaluate Fine-Tuned Model
# ------------------------------------------------------------

fine_tune_loss, fine_tune_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print()
print("Fine-Tuning Results")
print("-------------------")
print(f"Validation Loss: {fine_tune_loss:.4f}")
print(f"Validation Accuracy: {fine_tune_accuracy:.4f}")


# ------------------------------------------------------------
# Step 20 - Plot Fine-Tuning History
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    fine_tune_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    fine_tune_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Fine-Tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    fine_tune_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    fine_tune_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Fine-Tuning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Step 21 - Final Confusion Matrix
# ------------------------------------------------------------

y_true = []
y_pred = []

for images, labels in val_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype("int32")

    y_true.extend(
        labels.numpy()
        .flatten()
        .astype("int32")
    )

    y_pred.extend(
        predictions.flatten()
    )


y_true = tf.convert_to_tensor(y_true)
y_pred = tf.convert_to_tensor(y_pred)


final_confusion_matrix = tf.math.confusion_matrix(
    y_true,
    y_pred
)

print()
print("Final Confusion Matrix")
print("----------------------")
print(final_confusion_matrix.numpy())


# ------------------------------------------------------------
# Step 22 - Compare Transfer Learning and Fine-Tuning
# ------------------------------------------------------------

print()
print("Model Comparison")
print("----------------")

print(
    f"Transfer Learning Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Fine-Tuning Accuracy: "
    f"{fine_tune_accuracy:.4f}"
)

print()

print(
    f"Transfer Learning Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Fine-Tuning Loss: "
    f"{fine_tune_loss:.4f}"
)


# ------------------------------------------------------------
# Step 18 - Fine-Tune Model with Early Stopping
# ------------------------------------------------------------

# Stop training when validation loss stops improving.
# Restore the model weights from the best epoch.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

FINE_TUNE_EPOCHS = 10

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stopping]
)


# ------------------------------------------------------------
# Step 19 - Evaluate Fine-Tuned Model
# ------------------------------------------------------------

fine_tune_loss, fine_tune_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print()
print("Fine-Tuning Results")
print("-------------------")
print(f"Validation Loss: {fine_tune_loss:.4f}")
print(f"Validation Accuracy: {fine_tune_accuracy:.4f}")


# ------------------------------------------------------------
# Step 20 - Display Training History
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    fine_tune_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    fine_tune_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Fine-Tuning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    fine_tune_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    fine_tune_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Fine-Tuning Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Step 21 - Final Confusion Matrix
# ------------------------------------------------------------

y_true = []
y_pred = []

for images, labels in val_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype("int32")

    y_true.extend(
        labels.numpy()
        .flatten()
        .astype("int32")
    )

    y_pred.extend(
        predictions.flatten()
    )


y_true = tf.convert_to_tensor(y_true)
y_pred = tf.convert_to_tensor(y_pred)


final_confusion_matrix = tf.math.confusion_matrix(
    y_true,
    y_pred
)

print()
print("Final Confusion Matrix")
print("----------------------")
print(final_confusion_matrix.numpy())


# ------------------------------------------------------------
# Step 22 - Model Comparison
# ------------------------------------------------------------

print()
print("Model Comparison")
print("----------------")

print(
    f"Transfer Learning Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Fine-Tuning Accuracy: "
    f"{fine_tune_accuracy:.4f}"
)

print()

print(
    f"Transfer Learning Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Fine-Tuning Loss: "
    f"{fine_tune_loss:.4f}"
)

print()

print(
    "Fine-Tuning Epochs Completed:",
    len(fine_tune_history.history["loss"])
)
