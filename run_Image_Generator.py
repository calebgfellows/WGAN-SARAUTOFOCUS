import Image_Generator
import os

from PIL import Image
import numpy as np

inputsize = 32
def run():

    og_path = r'?\WGANAutofocus\Random\Train'
    image_path = og_path+"\Org"
    i = 0
    num = len(os.listdir(image_path))
    images = np.zeros((num,inputsize,inputsize))

    for img in os.listdir(image_path):

        image2 = Image.open(os.path.join(image_path, ("%s") % img))
        tempimage = np.array(image2)
        if tempimage.shape == (inputsize, inputsize, 3):
            tempimage = np.dot(tempimage, [0.299, 0.587, 0.114])
        images[i] = tempimage

        i = i + 1

    Image_Generator.run(images, 100000, og_path)


if __name__ == '__main__':
    run()
