"""
Run face detection on each images in `in_dir`, crop out largest detected face, save flattened images to csv specified by `out_file`.

Refer to: http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

"""

import numpy as np
import cv2
import os
from optparse import OptionParser
from utils import get_scores, vectorize_scores

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i",
                      dest="in_dir", type="string", default="data/gifs/test",
                      help="Load images from this directory")
    parser.add_option("-o",
                      dest="out_file", type="string", default="test.csv",
                      help="Save images to this file")
    (options, args) = parser.parse_args()


    face_cascade = cv2.CascadeClassifier('haar_data/haarcascade_frontalface_default.xml')


    scores = get_scores('data/json/', 'multi_hot')
    scores = vectorize_scores(scores)

    f = open(options.out_file, 'w')
    f.write('emotion,pixels\n')

    gifs = os.listdir(options.gifs_dir)

    for gif in gifs:
        frames = os.listdir(options.gifs_dir + '/' + gif)

        score = scores[gif]

        # Sample 10 equally-spaced frames if we have >10 frames
        lf = len(frames)
        if lf > 10:
            space = lf / 10
            frames = frames[::space]
            frames = np.random.choice(frames, 10, replace=False)

        for fname in frames:

            # Load image, convert to grayscale
            img = cv2.imread(options.gifs_dir + '/' + gif + '/' + fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ih, iw = gray.shape

            # Detect largest face
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                x = np.random.randint(0, iw - 48)
                y = np.random.randint(0, ih - 48)
                crop =  gray[y:y+48, y:y+48]

            else:
                x,y,w,h = sorted(faces, key=lambda F: -(F[2] * F[3]))[0] 

                # Crop detected face and write image to file
                # Crop as square by expanding face box in minimum dimension; handle borders
                if w == h:
                    crop = gray[y:y+h, x:x+w]
                elif w > h:
                    d = w - h
                    p1 = y - np.floor(d / 2.0).astype(int)
                    p2 = y + np.ceil(d / 2.0).astype(int) 
                    if p2 > ih:
                        p1 -= (p2 - ih)
                        p2 = ih
                    elif p1 < 0:
                        p2 += abs(p1)
                        p1 = 0
                    crop =  gray[p1:p2, x:x+w]
                elif h > w:
                    d = h - w
                    p1 = x - np.floor(d / 2.0).astype(int)
                    p2 = x + np.ceil(d / 2.0).astype(int) 
                    if p2 > iw:
                        p1 -= (p2 - iw)
                        p2 = iw
                    elif p1 < 0:
                        p2 += abs(p1)
                        p1 = 0
                    crop =  gray[y:y+h, p1:p2]

                scale_factor = max(w,h) / 48.0
                crop = cv2.resize(crop, None, fx=scale_factor, fy=scale_factor)

            if crop.shape != (48,48): 
                print 'ERROR: bad crop shape: {}'.format(crop.shape); break

            img_str = ' '.join([str(i) for i in crop.flatten()])

            # always write 0 as emotion since we don't care about this anyway
            f.write(score.tostring() + ',' + img_str + '\n')


