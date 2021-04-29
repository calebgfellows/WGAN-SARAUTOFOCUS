
import WGAN
import os

import numpy as np
import cv2
import Load_Datasets
checkpoint_path = "Random.h5"
Generator = WGAN.Image_Gen()

Generator.load_weights(checkpoint_path)

# width and height of the images
datasize = 32
path = r'?\WGANAutofocus' # Replace ? with location of WGANAutofocus



batch_size = "length"
patch_size = (datasize,datasize)
train_dataset, train_dataset_length = Load_Datasets.load(
    "Random",
    mode="Test",
    batch_size="length",
    patch_size=patch_size,
    shuffle=False,
    path=path
)

train_np = np.stack(list(train_dataset))

predictions = []
i = 100000
for x in range(train_dataset_length):
    prediction = train_np[x][1]
    prediction = prediction.reshape(1, datasize, datasize, 3).astype('float32')
    prediction = (Generator(prediction, training=False)).numpy()
    prediction = (prediction*127.5)+127.5
    savepath = path + "\Results"

    savenumber = i+x
    cv2.imwrite(os.path.join(savepath, ("%d.png") % savenumber), prediction[0])


