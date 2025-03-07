import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


img_height =128
img_width = 128
batch_size =32

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Define a simple CNN model
def build_eczema_cnn(input_shape=(128, 128, 3)):
    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification (eczema or not)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Instantiate the model
model = build_eczema_cnn()

# Summary of the model
model.summary()

# Load training and validation data
train_dir = 'dataset/train_data_4'

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Load test data
test_dir = "dataset/test_data"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',  # Assign labels based on folder names
    label_mode='binary',  # Binary classification (eczema vs. non-eczema)
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Optimize test dataset
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    steps_per_epoch=len(train_ds),
    validation_steps=len(val_ds)
)

# Get true labels from the validation dataset
y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)

# Get predictions from the model
y_pred_probs = model.predict(val_ds)  # Returns probabilities
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert to binary labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Eczema"], yticklabels=["Normal", "Eczema"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Validation Data")


# Evaluate model on the test dataset
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test Accuracy: {test_acc:.2f}")

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["Normal", "Eczema"]))

plt.savefig('Results/Confusion_Matricies/images1')
plt.show()

# Get true labels and images from validation dataset
y_true = []
images = []
for img_batch, label_batch in val_ds:
    images.extend(img_batch.numpy())  # Convert images to NumPy arrays
    y_true.extend(label_batch.numpy())  # Convert labels to NumPy arrays

y_true = np.array(y_true)

# Get predictions
y_pred_probs = model.predict(val_ds)  # Predict probabilities
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert to binary predictions

# Identify misclassified images
misclassified_indices = np.where(y_true != y_pred)[0]
misclassified_images = [images[i] for i in misclassified_indices]
misclassified_labels = [y_true[i] for i in misclassified_indices]
misclassified_preds = [y_pred[i] for i in misclassified_indices]

# Plot misclassified images
num_images = len(misclassified_images)
cols = 5
rows = (num_images // cols) + 1  # Ensure enough rows

plt.figure(figsize=(15, rows * 3))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(misclassified_images[i].astype("uint8"))  # Convert back to image format
    plt.title(f"True: {int(misclassified_labels[i])} | Pred: {int(misclassified_preds[i])}", fontsize=10)
    plt.axis("off")

show = input("See incorrectly prdicted images? !Warning! very graphic (y/n): ")
if show == 'y':
    plt.savefig("Results/wrong_predictions/images1")
    plt.tight_layout()
    plt.show()
else:
    pass

