import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

class Predictor(object):
    def __init__(self, graph_path, model_path):
        ## Let us restore the saved model
        sess = tf.Session()
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph(graph_path)
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")

        self._sess = sess
        self._y_pred = y_pred
        self._x = x
        self._y_true = y_true


    def predictClass(self, frame, num_classes):
        height, width, num_channels = frame.shape
        images = []
        images.append(frame)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        x_batch = images.reshape(1, height, width, num_channels)
        y_test_images = np.zeros((1, num_classes))

        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {self._x: x_batch, self._y_true: y_test_images}
        result=self._sess.run(self._y_pred, feed_dict=feed_dict_testing)
        result = np.argmax(result, axis=1)
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        return result[0]
