import keras.layers
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

#혼동행렬 추가
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

#사전학습
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input



IMAGE_SIZE = (128,128)

def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print(class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


# Load image data
def load_test_data(folder_path):
    X = []
    filenames = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            filenames.append(image_name)
    X = np.array(X)
    return X, filenames

# # Load training and testing data
train_folder = r"C:\Users\AMD\PycharmProjects\PythonProject1\data\archive\train"
test_folder = r"C:\Users\AMD\PycharmProjects\PythonProject1\data\archive\test"
X_train, y_train, class_names = load_train_data(train_folder)
X_test, test_filenames = load_test_data(test_folder)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize values
X_train = preprocess_input(X_train.copy())
X_test = preprocess_input(X_test.copy())

print("X_train_split.shape:", X_train.shape)
print("y_train_split.shape:", y_train.shape)
print("X_test_split.shape:", X_test.shape)
print("y_test_split.shape:", y_test.shape)

import cv2

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    temp_img = (X_train[i] + 1) * 127.5
    temp_img = temp_img.astype(np.uint8)

    rgb_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_img)
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

#X_train.astype(np.float32)/255.0
#X_test=X_test.astype(np.float32)/255.0




#여기가 해야될 부분!!=============================

# 라벨 원핫인코딩
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# 모델 구성 및 학습
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(128,128,3))
resnet_base.trainable = False

model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-4),
    metrics=['accuracy']
)

hist = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

#혼동행렬 추가
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)



#acc, loss, 혼동행렬
import matplotlib.pyplot as plt

#acc
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

#loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()

#혼동행렬
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=True)
plt.title("Confusion Matrix")
plt.show()
