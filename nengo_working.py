from urllib.request import urlopen
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.slim as slim
import nengo
import nengo_dl

IMAGE_DIMS = (96, 96, 3)
from tensorflow.python.keras.models import load_model, clone_model
#from keras.models import load_model,clone_model
from tensorflow.python.keras.preprocessing.image import img_to_array

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from nengo_classification.pyimagesearch.smallervggnet import SmallerVGGNet
from imutils import paths
import argparse
import random
import pickle
import h5py

import imutils

from tensorflow.python.tools import inspect_checkpoint as chkp


def three():
    K.clear_session()
    data = []
    """
    labels = [['black', 'jeans']]*344 + [['blue', 'dress']]*386 + [['blue', 'jeans']]*356 + [['blue', 'shirt']]*369 + \
             [['blue', 'sweater']]*99 + [['gray', 'shorts']]*96 + [['red', 'dress']]*380 + [['red', 'shirt']]*332
    """
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
    IMAGE_DIMS = (96, 96, 3)



    checkpoints_dir = '.\\checkpoints'
    # load the image, pre-process it, and store it in the data list
    import glob
    image_types = ('*.jpg', '*.jpeg', '*.png')
    files = []
    labels = []
    folders = glob.glob("D:\\Users\\bob\\PycharmProjects\\test_recommendation\\nengo_classification\\dataset\\*\\")
    for image_type in image_types:
        files.extend(glob.glob("D:\\Users\\bob\\PycharmProjects\\test_recommendation\\nengo_classification\\dataset\\*\\" + image_type))
    print(len(files))
    for f in files:
        """
        image = cv2.imread(f)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        """
        label = f.split("\\")[-2].split("_")
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.load("data.npy")
    labels = np.array(labels)

    # print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

    # binarize the labels using scikit-learn's special multi-label
    # binarizer implementation
    # print("[INFO] class labels:")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    #print(labels)

    """
    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i + 1, label))

    print(data.shape)
    print(labels.shape)
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.2, random_state=42)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
    model = SmallerVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
        finalAct="sigmoid")

    # initialize the optimizer (SGD is sufficient)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training network...")
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save("fashion_model.h5")
    model.save_weights("fashion_model_weights.h5")

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig("plot.png")
    """
    class KerasNode:
        def __init__(self, keras_model, mlb):
            self.model = keras_model
            self.mlb = mlb

        def pre_build(self, *args):
            self.model = clone_model(self.model)

        def __call__(self, t, x):
            # pre-process the image for classification
            img = tf.reshape(x, (-1,) + IMAGE_DIMS)
            #print(img.shape)
            """
            img = cv2.resize(img, (96, 96))
            img = img.astype("float") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            """
            return self.model.call(img)
            #eturn self.model.call(tf.convert_to_tensor(img, dtype=tf.float32))

        def post_build(self, sess, rng):
            with sess.as_default():
                self.model.load_weights("fashion_model_weights.h5")
                self.mlb = pickle.loads(open("mlb.pickle", "rb").read())
            #pass

    net_input_shape = np.prod((96, 96, 3))  # because input will be a vector

    with nengo.Network() as net:
        # create a normal input node to feed in our test image.
        # the `np.ones` array is a placeholder, these
        # values will be replaced with the Fashion MNIST images
        # when we run the Simulator.
        input_node = nengo.Node(output=np.ones((net_input_shape,)))

        # create a TensorNode containing the KerasNode we defined
        # above, passing it the Keras model we created.
        # we also need to specify size_in (the dimensionality of
        # our input vectors, the flattened images) and size_out (the number
        # of classification classes output by the keras network)
        model = load_model("fashion_model.h5")
        mlb = pickle.loads(open("mlb.pickle", "rb").read())
        keras_node = nengo_dl.TensorNode(
            KerasNode(model, mlb),
            size_in=net_input_shape,
            size_out=len(mlb.classes_))

        # connect up our input to our keras node
        nengo.Connection(input_node, keras_node, synapse=None)

        # add a probes to collect output of keras node
        keras_p = nengo.Probe(keras_node)
        input_p = nengo.Probe(input_node)

    minibatch_size = 20

    np.random.seed(3)
    test_inds = np.random.randint(low=0, high=data.shape[0],
                                  size=(minibatch_size,))
    test_inputs = data[test_inds]

    # flatten images so we can pass them as vectors to the input node
    test_inputs = test_inputs.reshape((-1, net_input_shape))

    # unlike in Keras, NengoDl simulations always run over time.
    # so we need to add the time dimension to our data (even though
    # in this case we'll just run for a single timestep).
    test_inputs = test_inputs[:, None, :]

    with nengo_dl.Simulator(net, minibatch_size=len(test_inputs)) as sim:
        sim.step(data={input_node: test_inputs})

    tensornode_output = sim.data[keras_p]

    for i in range(len(test_inputs)):
        plt.figure()
        b, g, r = cv2.split(data[test_inds[i]])
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img)
        print("[INFO] classifying ...")
        proba = tensornode_output[i][0]
        print(proba)
        idxs = np.argsort(proba)[::-1]
        print(idxs)
        print(np.argmax(tensornode_output[i, 0]))
        plt.axis("off")
        plt.title("%s, %s" % (mlb.classes_[idxs[0]], mlb.classes_[idxs[1]]))
        plt.show()


three()
