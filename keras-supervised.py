import pandas as pd
import tensorflow as tf
from tensorflow import keras

base_dir_1 = "db/train"
# val_dir = "/db/val"
# test_dir = "/db/test"  

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
])

train_ds = keras.utils.image_dataset_from_directory(
    base_dir_1,
    image_size=(200,200),
    batch_size=128,
    seed=100,
    subset='training',
    validation_split=0.2
)

class_names = train_ds.class_names
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

val_ds = keras.utils.image_dataset_from_directory(
    base_dir_1,
    seed=1,
    image_size=(200,200),
    batch_size=128,
    subset='validation',
    validation_split=0.06
)

test_ds = keras.utils.image_dataset_from_directory(
    base_dir_1,
    seed=2,
    image_size=(200,200),
    batch_size=128,
    subset='validation',
    validation_split=0.14
)

num_classes = len(class_names)
 
model = keras.models.Sequential([
    keras.layers.Rescaling(1./255, input_shape=(200,200,3)),
    keras.layers.Conv2D(filters=16, kernel_size=(3,3),padding='valid',activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=32, kernel_size=(4,4),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(4,4)),
    keras.layers.Conv2D(filters=32, kernel_size=(5,5),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(4,4)),
    keras.layers.Conv2D(filters=64, kernel_size=(5,5),padding='same',activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(3,3)),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=32,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=256,activation='relu'),
    keras.layers.Dense(units=512,activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(train_ds,epochs=20,validation_data=val_ds)
print("Training finished")

# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test accuracy: {test_acc}")

model.fit(train_ds,epochs=10,validation_data=val_ds)
print("Training finished")

model.fit(train_ds,epochs=15,validation_data=val_ds)
print("Training finished")

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")