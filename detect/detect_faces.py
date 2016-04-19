"""
Run face detection on each images in `in_dir`, crop out largest detected face, save each image to file in `out_dir`.

Note
----
saves image as grayscale

TODO
----
should this be cropped to 48 x 48 image? this is what Tang's network was trained on . . .

"""

import numpy as np
import cv2
import os
from optparse import OptionParser

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i",
                      dest="in_dir", type="string", default="./in_imgs",
                      help="Load images from this directory")
    parser.add_option("-o",
                      dest="out_dir", type="string", default="./out_imgs",
                      help="Save images to this directory")
    (options, args) = parser.parse_args()

    if not os.path.isdir(options.out_dir): os.mkdir(options.out_dir)

    face_cascade = cv2.CascadeClassifier('haar_data/haarcascade_frontalface_default.xml')


    for fname in os.listdir(options.in_dir):

        # Load image, convert to grayscale
        img = cv2.imread(options.in_dir + '/' + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect largest face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        x,y,w,h = sorted(faces, key=lambda F: -(F[2] * F[3]))[0] 

        # Crop detected face and write image to file
        cv2.imwrite(options.out_dir + '/' + fname, gray[y:y+h, x:x+w])