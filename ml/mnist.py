import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[:, ..., np.newaxis] / 255.0
X_test = X_test[:, ..., np.newaxis] / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callbacks = [tf.keras.callbacks.ModelCheckpoint("model_best.h5")]

model.fit(
    X_train, y_train, validation_split=0.2, callbacks=callbacks, epochs=1, batch_size=32
)
model.load_weights("model_best.h5")
y_test_hat = model.predict(X_test)
print(accuracy_score(np.argmax(y_test_hat, axis=1), y_test))
