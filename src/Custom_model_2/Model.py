from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, stride=1):
    shortcut = x
    
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def depthwise_block(x, filters, stride=1):
    shortcut = x

    x = layers.DepthwiseConv2D((3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (1,1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and shortcut.shape[-1] == filters:
        x = layers.Add()([x, shortcut])

    x = layers.ReLU()(x)

    return x


def build_model():
    inputs = keras.Input(shape=(224, 224, 3))

    # Initial Stem
    x = layers.Conv2D(32, (3,3), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1
    x = conv_block(x, 64, stride=1)
    x = conv_block(x, 64, stride=1)

    # Stage 2
    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128, stride=1)

    # Stage 3 (MobileNet style)
    x = depthwise_block(x, 256, stride=2)
    x = depthwise_block(x, 256, stride=1)

    # Stage 4
    x = depthwise_block(x, 512, stride=2)
    x = depthwise_block(x, 512, stride=1)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(69, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model