from queue import Queue
import tensorflow as tf
import numpy as np
import centroidTracker
import threading
from preModel import build_siamese_model
from preModel import euclidean_distance
import warnings
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings

# path

VIDEO_PATH = "data/videos/video3.mp4"
modelCFG = "YOLOv3/YL/config.cfg"
modelWeights = "YOLOv3/YL/yolov3-spp.weights"
PATH_CPK = "SIAMESE/myModel/siamese_model/checkpoints/"
IMG_SHAPE = (84, 33, 3)

# global vars
global video, vcache, vcachel, counterl, counter, verify, ct, stop, net, outputNames, rfrs, pt1s, pt2s, predictions

globals()["ct"] = centroidTracker.CentroidTracker()
globals()["video"] = cv2.VideoCapture(VIDEO_PATH)
globals()["vcachel"] = Queue(maxsize=40)
globals()["vcache"] = Queue(maxsize=60)
globals()["verify"] = False
globals()["stop"] = False
globals()["counter"] = 0
globals()["counterl"] = 0
globals()["pt1s"] = []
globals()["pt2s"] = []
globals()["rfrs"] = []
globals()["predictions"] = []

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


def videocache():
    while True:
        if globals()["counter"] < 60:
            globals()["counter"] = globals()["counter"] + 1
            threading.Thread(target=frameloader, args=()).start()
        if globals()["stop"] == True:
            break
    return

def frameloader():
    while True:
        if globals()["verify"] == False:
            globals()["verify"] = True
            (grab, quadro) = globals()["video"].read()
            globals()["verify"] = False
            break
    if not grab:
        globals()['stop'] = True
        return
    globals()['vcache'].put(quadro)
    return

def videocache1():
    while True:
        if globals()["counterl"] < 30:
            print(globals()["counterl"])
            globals()["counterl"] = globals()["counterl"] + 10
            process()
        if globals()["stop"] == True:
            break
    return

def process():
    frame = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame1 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame2 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame3 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame4 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame5 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame6 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame7 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame8 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    frame9 = globals()['vcache'].get()
    globals()['counter'] = globals()['counter'] - 1
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(globals()["outputNames"])
    hT, wT, cT = frame.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidance = scores[classId]
            if confidance > 0.6 and classId == 0:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidance))

    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.6, 0.3)

    rects = []

    for i in indices:

        rects.append([bbox[i][0], bbox[i][1], bbox[i][0] + bbox[i][2], bbox[i][1] + bbox[i][3]])

    objects = globals()["ct"].update(rects)


    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame

        if objectID in pt1s:

            cv2.putText(frame, "pt1", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame1, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame2, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame3, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame3, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame4, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame4, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame5, "pt1", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame5, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame6, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame6, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame7, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame7, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame8, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame8, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame9, "pt1", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame9, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        elif objectID in pt2s:

            cv2.putText(frame, "pt2", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame1, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame2, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame2, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame3, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame3, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame4, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame4, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame5, "pt2", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame5, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame6, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame6, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame7, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame7, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame8, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame8, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
            cv2.putText(frame9, "pt2", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame9, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
        elif objectID in rfrs:

            cv2.putText(frame, "rfr", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame1, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame2, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame3, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame3, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame4, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame4, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame5, "rfr", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame5, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame6, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame6, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame7, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame7, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame8, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame8, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame9, "rfr", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame9, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        else:

            for j in rects:

                if centroid[0] > j[0] and centroid[0] < j[2] and centroid[1] > j[1] and centroid[1] < j[3]:

                    k = j

                    pt = frame[k[1]:k[3], k[0]:k[2]].copy()

                    ptr = cv2.resize(pt, (33, 84)).reshape((-1, 3))

                    # convert to np.float32
                    ptr = np.float32(ptr)

                    # define criteria, number of clusters(K) and apply kmeans()
                    ret, label, center = cv2.kmeans(ptr, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                    # Now convert back into uint8, and make original image
                    center = np.uint8(center)

                    res = center[label.flatten()]

                    res0 = res.reshape(cv2.resize(pt, (33, 84)).shape)

                    # add channel a dimension to both the images
                    res0 = np.expand_dims(res0, axis=-1)

                    # add a batch dimension to both images
                    res0 = np.expand_dims(res0, axis=0)

                    # scale the pixel values to the range of [0, 1]
                    res0 = res0 / 255.0

                    predict1 = model.predict([res21, res0])

                    if predict1[0][0] > 0.7:
                        globals()["pt1s"].append(objectID)
                        cv2.putText(frame, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame1, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame2, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame3, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame3, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame4, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame4, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame5, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame5, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame6, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame6, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame7, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame7, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame8, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame8, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame9, "pt1", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame9, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


                    predict2 = model.predict([res22, res0])

                    if predict2[0][0] > 0.7:
                        globals()["pt2s"].append(objectID)
                        cv2.putText(frame, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame1, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame1, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame2, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame2, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame3, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame3, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame4, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame4, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame5, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame5, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame6, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame6, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame7, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame7, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame8, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame8, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
                        cv2.putText(frame9, "pt2", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame9, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)


                    predict3 = model.predict([res23, res0])

                    if predict3[0][0] > 0.85:
                        globals()["rfrs"].append(objectID)
                        cv2.putText(frame, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame1, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame2, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame3, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame3, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame4, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame4, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame5, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame5, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame6, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame6, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame7, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame7, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame8, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame8, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                        cv2.putText(frame9, "rfr", (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(frame9, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

                    break


    globals()['vcachel'].put(frame)
    globals()['vcachel'].put(frame1)
    globals()['vcachel'].put(frame2)
    globals()['vcachel'].put(frame3)
    globals()['vcachel'].put(frame4)
    return



# first frame process

# load YOLOv3 ssp network

net = cv2.dnn.readNetFromDarknet(modelCFG, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# prepare frame data

(grabed, frame) = globals()["video"].read()
blob = cv2.dnn.blobFromImage(frame, 1/255, (608, 608), [0, 0, 0], 1, crop=False)

# compute detections

net.setInput(blob)
layersNames = net.getLayerNames()
globals()["outputNames"] = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(globals()["outputNames"])

hT, wT, cT = frame.shape
bbox = []
classIds = []

confs = []
for output in outputs:
    for det in output:
        scores = det[5:]
        classId = np.argmax(scores)
        confidance = scores[classId]
        if confidance > 0.3 and classId == 0:
            w, h = int(det[2]*wT), int(det[3]*hT)
            x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
            bbox.append([x, y, w, h])
            classIds.append(classId)
            confs.append(float(confidance))

indices = cv2.dnn.NMSBoxes(bbox, confs, 0.3, 0.4)

xmin = 3000
xmax = 0
distcenter = []

for i in indices:
    i = i

    dist = (((bbox[i][1]+(bbox[i][3]/2)) - hT)**2)**0.5

    distcenter.append(dist)

    if bbox[i][0] < xmin:

        pmin = bbox[i]
        xmin = bbox[i][0]

    if bbox[i][0] > xmax:

        pmax = bbox[i]
        xmax = bbox[i][0]


# define team members

player_team_1 = frame[pmin[1]:pmin[1]+pmin[3],pmin[0]:pmin[0]+pmin[2]].copy()
player_team_2 = frame[pmax[1]:pmax[1]+pmax[3],pmax[0]:pmax[0]+pmax[2]].copy()

player_team_1r = cv2.resize(player_team_1, (33, 84)).reshape((-1, 3))
player_team_2r = cv2.resize(player_team_2, (33, 84)).reshape((-1, 3))

# convert to np.float32
player_team_1r = np.float32(player_team_1r)
player_team_2r = np.float32(player_team_2r)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret1, label1, center1 = cv2.kmeans(player_team_1r, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
ret2, label2, center2 = cv2.kmeans(player_team_2r, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center1 = np.uint8(center1)
center2 = np.uint8(center2)

res1 = center1[label1.flatten()]
res2 = center2[label2.flatten()]

res21 = res1.reshape(cv2.resize(player_team_1, (33, 84)).shape)
res22 = res2.reshape(cv2.resize(player_team_2, (33, 84)).shape)


# add channel a dimension to both the images
res21 = np.expand_dims(res21, axis=-1)
res22 = np.expand_dims(res22, axis=-1)

# add a batch dimension to both images
res21 = np.expand_dims(res21, axis=0)
res22 = np.expand_dims(res22, axis=0)

# scale the pixel values to the range of [0, 1]
res21 = res21 / 255.0
res22 = res22 / 255.0

distmin = 3000

f = 0

for i in indices:

    i = i

    centery = bbox[i][1] + (bbox[i][3]/2)

    pt = frame[bbox[i][1]:bbox[i][1] + bbox[i][3], bbox[i][0]:bbox[i][0] + bbox[i][2]].copy()

    ptr = cv2.resize(pt, (33, 84)).reshape((-1, 3))

    # convert to np.float32
    ptr = np.float32(ptr)

    # define criteria, number of clusters(K) and apply kmeans()
    ret, label, center = cv2.kmeans(ptr, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    res = center[label.flatten()]

    res0 = res.reshape(cv2.resize(pt, (33, 84)).shape)

    # add channel a dimension to both the images
    res0 = np.expand_dims(res0, axis=-1)

    # add a batch dimension to both images
    res0 = np.expand_dims(res0, axis=0)

    # scale the pixel values to the range of [0, 1]
    res0 = res0 / 255.0

    predict1 = model.predict([res21, res0])
    predict2 = model.predict([res22, res0])

    if predict1 < 0.5 and predict2 < 0.5 and distcenter[f] < distmin and centery > (hT*0.3) and centery < (hT-(hT*0.3)):
        aridx = i
        distmin = distcenter[f]

    f = f + 1

arbitro = frame[bbox[aridx][1]:bbox[aridx][1] + bbox[aridx][3], bbox[aridx][0]:bbox[aridx][0] + bbox[aridx][2]].copy()

arbtr = cv2.resize(arbitro, (33, 84)).reshape((-1, 3))

# convert to np.float32
arbtr = np.float32(arbtr)

# define criteria, number of clusters(K) and apply kmeans()
ret3, label3, center3 = cv2.kmeans(arbtr, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center3 = np.uint8(center3)

res3 = center3[label3.flatten()]

res23 = res3.reshape(cv2.resize(arbitro, (33, 84)).shape)

# add channel a dimension to both the images
res23 = np.expand_dims(res23, axis=-1)

# add a batch dimension to both images
res23 = np.expand_dims(res23, axis=0)

# scale the pixel values to the range of [0, 1]
res23 = res23 / 255.0


threading.Thread(target=videocache, args=()).start()

while True:
    if globals()["counter"] == 60:
        break

threading.Thread(target=videocache1, args=()).start()

while True:
    if globals()["counterl"] == 30:
        break

while True:
    frame = globals()["vcachel"].get()
    frame1 = globals()["vcachel"].get()
    frame2 = globals()["vcachel"].get()
    frame3 = globals()["vcachel"].get()
    frame4 = globals()["vcachel"].get()
    frame5 = globals()["vcachel"].get()
    frame6 = globals()["vcachel"].get()
    frame7 = globals()["vcachel"].get()
    frame8 = globals()["vcachel"].get()
    frame9 = globals()["vcachel"].get()
    globals()["counterl"] = globals()["counterl"] - 10
    cv2.imshow("Rp", frame)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame1)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame2)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame3)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame4)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame5)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame6)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame7)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame8)
    cv2.waitKey(0)
    cv2.imshow("Rp", frame9)
    cv2.waitKey(0)


