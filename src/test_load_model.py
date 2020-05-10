import tensorflow as tf
import numpy as np
import skimage.transform

from CNN_Model import normalize_im
from dataimport import readStackFromTiff
import cv2
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

matList = readStackFromTiff("/home/gidi/src/deep-storm/data/testStack_SimulatedMicrotubules.tif")
path = "/home/gidi/src/deep-storm/data/model"
data =matList[0]
data = skimage.transform.resize(data,(208,208))
data = data.reshape((208,208,1))
output_layer = 'Prediction/convolution:0'
input_node = "input_1:0"
data=normalize_im(data, 0.1888016, 0.18061188)
data = data.reshape((208,208,1))
with tf.Session(config=config) as sess:
    imported = tf.saved_model.load(sess=sess,tags=[tf.saved_model.tag_constants.SERVING],export_dir=path)
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    predictions, = sess.run(prob_tensor, {input_node: [data],'BN-CNNF1/keras_learning_phase:0':True})
    print(predictions)

cv2.imshow("input",cv2.convertScaleAbs(data))
plt.imshow(data.reshape(208,208),cmap="gray")
cv2.imshow("pred",predictions)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
