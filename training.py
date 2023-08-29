import wandb
import numpy as np
import tensorflow as tf
keras = tf.keras

wandb.init(
    project='MNIST-Classifier'
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('mnist.npz')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

x_train = np.float32(x_train) / 255
x_test = np.float32(x_test) / 255

# adding 10000 test examples
x_test = np.concatenate((x_test, x_train[:10000]))
y_test = np.concatenate((y_test, y_train[:10000]))
x_train = x_train[10000:]
y_train = y_train[10000:]

# Expand the last dimension for the augmentation layers
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Create a tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Define the augmentation
data_augmentation = tf.keras.Sequential([
    keras.layers.experimental.preprocessing.RandomTranslation(height_factor=.6, width_factor=.6)
])

# Apply the augmentation only on 50% of the data
augmented_train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.concatenate(augmented_train_dataset)
train_dataset = train_dataset.shuffle(x_train.shape[0]*2)
train_dataset = train_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

# model = tf.keras.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation=keras.layers.LeakyReLU()),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(256, activation=keras.layers.LeakyReLU()),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(128, activation=keras.layers.LeakyReLU()),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(10, activation='softmax')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape inputs to (28, 28, 1)
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=keras.layers.LeakyReLU()),  # 32 filters of size 3x3
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU()),  # 64 filters of size 3x3
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation=keras.layers.LeakyReLU()),  # 128 filters of size 3x3
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=keras.layers.LeakyReLU()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

with tf.device('/GPU:0'):
    history = model.fit(train_dataset, epochs=25, validation_data=(x_test, y_test),
                    callbacks=[
                        wandb.keras.WandbMetricsLogger(log_freq=5),
                        wandb.keras.WandbModelCheckpoint('models')
                    ])

model.save('trained models/mlp_augmented.md5')
