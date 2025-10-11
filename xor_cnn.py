import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

xor = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
target_data = np.array([[0], [1], [1], [0]], dtype="float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

model.fit(train_data, target_data, epochs=1000, verbose=1)

loss, acc = model.evaluate(train_data, target_data, verbose=0)
print("loss:", round(loss, 6), "acc:", round(acc, 3))

pred = model.predict(train_data)
print("*" * 50)
print(np.round(pred, 3))
