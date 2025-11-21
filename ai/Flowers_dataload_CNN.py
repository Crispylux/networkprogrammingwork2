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
X_train = X_train / 255.0
X_test = X_test / 255.0

print("X_train_split.shape:", X_train.shape)
print("y_train_split.shape:", y_train.shape)
print("X_test_split.shape:", X_test.shape)
print("y_test_split.shape:", y_test.shape)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i]])  # 이미지에 해당하는 클래스 이름 표시
    plt.axis('off')
plt.show()

#X_train.astype(np.float32)/255.0
#X_test=X_test.astype(np.float32)/255.0




#여기가 해야될 부분!!=============================

# 라벨 원핫인코딩
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# 모델 구성 및 학습(CNN)
model = keras.models.Sequential([
    #특징 추출
    keras.layers.Conv2D(input_shape = (128, 128, 3),                                #128 128 3
                        kernel_size = (3, 3),padding='same',filters = 32),          #32개
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Conv2D(kernel_size = (3, 3), padding = 'same', filters = 64),      #64개
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Conv2D(kernel_size = (3, 3), padding = 'same', filters = 128),     #128개
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Flatten(),                                                         #평탄

    #분류
    keras.layers.Dense(128, activation='relu'),                                #128개
    keras.layers.Dense(5, activation='softmax')                                #최종 5개 분류되게
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




# acc, loss 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()
