from collections import defaultdict
from glob import glob
from random import choice, sample

import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



train_file_path = "/home/star/Desktop/kinship/train_ds.csv"
train_folders_path = "/home/star/Desktop/kinship/train/train-faces/"
val_famillies = "F09"

all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

#all_images
#print(all_images)

relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values, relationships.relationship.values))
relationships = [(x[0],x[1],x[2]) for x in relationships if x[0][:10] in ppl and x[1][:10] in ppl]

train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


from keras.preprocessing import image
def read_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


def gen(list_tuples, person_to_images_map, batch_size=8):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size)
        
        # All the samples are taken from train_ds.csv, labels are in the labels column
        labels = []
        for tup in batch_tuples:
          labels.append(tup[2])

        X1 = [x[0] for x in batch_tuples]
        X1 = np.array([read_img(train_folders_path + x) for x in X1])

        X2 = [x[1] for x in batch_tuples]
        X2 = np.array([read_img(train_folders_path + x) for x in X2])

        yield [X1, X2], np.array(labels)


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


file_path = "/home/star/Desktop/kinship/vgg_face.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

model = baseline_model()
# model.load_weights(file_path)
model.fit_generator(gen(train, train_person_to_images_map, batch_size=8),  validation_data=gen(val, val_person_to_images_map, batch_size=8), epochs=120,verbose=1, callbacks=callbacks_list, steps_per_epoch=150, validation_steps=100)

test_path = "/home/star/Desktop/kinship/test/"


submission = pd.read_csv('/home/star/Desktop/kinship/test_ds.csv')
predictions = []

for i in range(0, len(submission.p1.values), 32):
    X1 = submission.p1.values[i:i+32]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = submission.p2.values[i:i+32]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred
    
d = {'index': np.arange(0, 3000, 1), 'label':predictions}
submissionfile = pd.DataFrame(data=d)
submissionfile = submissionfile.round()
submission.to_csv("vgg_face.csv", index=False)

submissionfile.astype("int64").to_csv("/home/star/Desktop/kinship/xy2472.csv", index=False)
