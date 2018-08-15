from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import shutil
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


#============== An Self-Costumized SGDMomentumOptimizer =======================================
#refer to https://medium.com/bigdatarepublic/custom-optimizer-in-tensorflow-d5b41f75644a
class ProjectOptimizer(optimizer.Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta=0.5,
                 use_locking=False,
                 name="ProjectOptimizer"):
        super(ProjectOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta = beta

        self._lr_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  #cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(beta_t * m + lr_t * grad)
        var_update = state_ops.assign_sub(var, m_t)

        return control_flow_ops.group(*[var_update, m_t])


#======= A Model of CAE indicating the layers and their interconnection =====================
#refer to https://www.tensorflow.org/get_started/custom_estimators
def cnn_model_fn(features, labels, mode, params):
    """
        NOTE for Xavier uniform initializer: 
             the default initializer of layers in Tensorflow is glorot_uniform_initializer
             Therefore, there is no need to set the initializer intentially for layers.
             refer to line232-235: 
             https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variable_scope.py
    """
    """
        Layers:
    """
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[5, 5],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[5, 5],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    conv4_flat = tf.reshape(conv4, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(
        inputs=conv4_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.45, training=mode == tf.estimator.ModeKeys.TRAIN)
        
    #According to project requirements, logstic is used to assess the quality of CNN
    logits = tf.layers.dense(inputs=dropout, units=47)
    
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

#======== Compute the accuracy and loss of the CNN model on the training dataset =========
    #Use tf.summary.scalar to log data for tensorboard figure generation
    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar("prediction_accuracy", accuracy[1])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = ProjectOptimizer(
            beta=params["B"], learning_rate=params["R"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":
        tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

#==================== Load training and eval data ===============================
    file_train0 = np.load("./data/data_classifier_train.npz")
    train_data0 = file_train0["x_train"].astype(np.float32)
    train_labels0 = file_train0["y_train"].astype(np.int32)
    file_test0 = np.load("./data/data_classifier_test.npz")
    eval_data0 = file_test0["x_test"].astype(np.float32)
    eval_labels0 = file_test0["y_test"].astype(np.int32)
    
#==================== Split the training dataset for 5-fold ======================
    train_data_array = np.array_split(train_data0, 5)
    train_labels_array = np.array_split(train_labels0, 5)
    
#======== Delete the diretory probabily generated by previous tests ==============
    for N_RR in range(5):
        for N_BB in range(5):
            for K in range(5):
                if os.path.isdir("./trained_model" + "%d" % K + "%d" % N_RR +
                                 "%d" % N_BB):
                    shutil.rmtree("./trained_model" + "%d" % K + "%d" % N_RR +
                                  "%d" % N_BB, True)
                                  
#=== Numerate the parameters, including learning_rate and beta for momentums =====
    N_RR = 0
    for RR in [0.0001, 0.001, 0.01]:
        N_BB = 0
        for BB in [0.1,0.2,0.5,0.8,1]:
            print("============== parameters ===============================" +
                  " R=%.4f" % RR + " B=%.4f" % BB)
            RR_BB_accuray = 0
            for K in range(5):
            
#========= Select one fold for evaluation while others for training =============
                if (K == 0):
                    train_data = np.concatenate(
                        train_data_array[1:5], axis=0).astype(np.float32)
                    train_labels = np.concatenate(
                        train_labels_array[1:5], axis=0).astype(np.int32)
                if (K == 4):
                    train_data = np.concatenate(
                        train_data_array[0:4], axis=0).astype(np.float32)
                    train_labels = np.concatenate(
                        train_labels_array[0:4], axis=0).astype(np.int32)
                if (K < 4 and K > 0):
                    train_data = np.concatenate(
                        np.concatenate(
                            (train_data_array[0:K], train_data_array[K + 1:5]),
                            axis=0),
                        axis=0).astype(np.float32)
                    train_labels = np.concatenate(
                        np.concatenate(
                            (train_labels_array[0:K],
                             train_labels_array[K + 1:5]),
                            axis=0),
                        axis=0).astype(np.int32)

                eval_data = train_data_array[K].astype(np.float32)
                eval_labels = train_labels_array[K].astype(np.int32)

#=========================  Create the estimator ================================
                prj_classifier = tf.estimator.Estimator(
                    model_fn=cnn_model_fn,
                    params={
                        "R": RR,
                        "B": BB
                    },
                    model_dir="./trained_model" + "%d" % K + "%d" % N_RR +
                    "%d" % N_BB,
                    config=tf.contrib.learn.RunConfig(num_cores=8))
                    
#======  Check accuracy and loss for every 8000 training samples ================
                evaluation_interval = 8000
                epoch_num = 2
                batch_size_user = 50
                steps_num = evaluation_interval / batch_size_user

                # Set up logging for predictions
                # Log the values in the "softmax_tensor" tensor with label "probabilities"
                # Log the values in the "train_accuracy" tensor with label "prediction_accuracy"
                tensors_to_log = {
                    "probabilities": "softmax_tensor",
                    "prediction_accuracy": "train_accuracy"
                }
                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=steps_num)
                tf.logging.set_verbosity(tf.logging.ERROR)

                for epoch in range(epoch_num):
#========================   Data Shuffling   ============================================
                    print("===============" + " R=%.4f" % RR + " B=%.4f" % BB +
                          " Ki=%d" % K + " epoch=%d" % epoch +
                          " =====================")
                    for i in range(
                            int(np.shape(train_data)[0] /
                                evaluation_interval)):
                        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                            x={
                                "x":
                                train_data[(i * evaluation_interval):((
                                    i + 1) * evaluation_interval)]
                            },
                            y=train_labels[(i * evaluation_interval):((
                                i + 1) * evaluation_interval)],
                            batch_size=batch_size_user,
                            num_epochs=None,
                            shuffle=True)
#========================   Execution of Training   =====================================
                        prj_classifier.train(
                            input_fn=train_input_fn,
                            steps=steps_num  
                        )

                        # Evaluate the model and print results
                        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                            x={"x": eval_data},
                            y=eval_labels,
                            num_epochs=1,
                            shuffle=False)
                        eval_results = prj_classifier.evaluate(
                            input_fn=eval_input_fn)
                        print(eval_results)
                RR_BB_accuray = RR_BB_accuray + eval_results["accuracy"]
                shutil.rmtree(
                    "./trained_model" + "%d" % K + "%d" % N_RR + "%d" % N_BB,
                    True)
            RR_BB_accuray = RR_BB_accuray / 5
            print("============== accuracy == for =======" + " Learning_rate=%.4f" % RR +
                  " Beta=%.4f" % BB + "is %.4f" % RR_BB_accuray)
            N_BB = N_BB + 1
        N_RR = N_RR + 1


if __name__ == "__main__":
    tf.app.run()
