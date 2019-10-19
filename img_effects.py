import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2

# tf.image.pad_to_bounding_box
# tf.image.crop_to_bounding_box
# tf.random_crop

# tf.

imname = './station.jpg'
im = tf.read_file(imname)
im_decoded = tf.image.decode_image(im)
im_decoded = tf.reshape(im_decoded,[1,422, 750, 3])
# resize_img = tf.image.resize_bicubic(im_decoded,[844,1500])
brightness_img = tf.image.adjust_brightness(im_decoded,-0.5)
sess = tf.Session()
im_decoded_val = sess.run(brightness_img)
im_decoded_val = im_decoded_val.reshape((422,750,3))
im_decoded_val = np.asarray(im_decoded_val,np.uint8)

img = cv2.cvtColor(im_decoded_val,cv2.COLOR_RGB2BGR)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()