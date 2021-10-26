import tensorflow as tf
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# paths

PATH_CPK = "SIAMESE/myModel/siamese_model/checkpoints/"
IMG_SHAPE = (84, 33, 3)

# siamese functions

def build_siamese_model(inputShape, embeddingDim=48):

    # specify the inputs for the feature extractor network
    inputs = tf.keras.layers.Input(inputShape)

    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # second set of CONV => RELU => POOL => DROPOUT layers
    x = tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # prepare the final outputs
    pooledOutput = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embeddingDim)(pooledOutput)

    # build the model
    model = tf.keras.models.Model(inputs, outputs)

    # return the model to the calling function
    return model

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1, keepdims=True)

    # return the euclidean distance between the vectors
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))

# load the pré trained model by the checkpoints for comparison
# configure the siamese network

imgA = tf.keras.layers.Input(shape=IMG_SHAPE)
imgB = tf.keras.layers.Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = tf.keras.layers.Lambda(euclidean_distance)([featsA, featsB])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
model = tf.keras.models.Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# compile the model
model.load_weights(PATH_CPK)
