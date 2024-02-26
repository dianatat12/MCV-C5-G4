import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    ReLU,
    Dense,
    GlobalAveragePooling2D,
    Add,
    SeparableConv2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    SGD,
    RMSprop,
    Adam,
    AdamW,
    Adadelta,
    Adagrad,
    Adamax,
    Adafactor,
    Nadam,
    Ftrl,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1


# model 1
import tensorflow as tf
from tensorflow.keras import layers



# model 11
class Model_11(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_11, self).__init__()
        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

    def call(self, x):
        return self.model(x)



def get_model(
    IMG_SIZE: int,
    IMG_CHANNEL: int,
    NUM_CLASSES: int,
    model_name: str,
    optimizer_name: str,
    lr: float = None,
    momentum: float = None,
):
    if optimizer_name == "SGD":
        optimizer = SGD()
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop()
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad()
    elif optimizer_name == "Adadelta":
        optimizer = Adadelta()
    elif optimizer_name == "Adam":
        optimizer = Adam()
    elif optimizer_name == "Adamax":
        optimizer = Adamax()
    elif optimizer_name == "Nadam":
        optimizer = Nadam()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    
    if model_name == "MODEL_11":
        model = Model_11(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )

    model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # total number of parameters
    total_params = model.count_params()
    # print(model.summary())
    return model, total_params
