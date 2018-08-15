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
    Encoder
    """
    
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # Now 28x28x1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)
    # Now 14x14x32
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)
    # Now 7x7x2
    encoded = tf.layers.conv2d(
        inputs=conv2,
        filters=2,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    """
    Decoder
    """

    upsample1 = tf.image.resize_images(
        encoded, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x2
    conv4 = tf.layers.conv2d(
        inputs=upsample1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    # Now 7x7x64
    upsample2 = tf.image.resize_images(
        conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 14x14x64
    conv5 = tf.layers.conv2d(
        inputs=upsample2,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    # Now 14x14x32
    upsample3 = tf.image.resize_images(
        conv5, size=[28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 28x28x32
    decodeds = tf.layers.conv2d(
        inputs=upsample3,
        filters=1,
        kernel_size=[5, 5],
        padding='same',
        activation=None)

    decodeds = tf.squeeze(decodeds, [3])
    # Now 28x28x1
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "MSE_training":
        tf.metrics.mean_squared_error(labels=labels, predictions=decodeds)
    }
    
    #According to project requirements, MSE is used to assess the quality of CAE
    MSE = tf.metrics.mean_squared_error(
        labels=labels, predictions=decodeds) 
        
#======== Compute the accuracy and loss of the CNN model on the training dataset =========
    #Use tf.summary.scalar to log data for tensorboard figure generation
    tf.identity(MSE[1], name="MSE_training_training")
    tf.summary.scalar("MSE_training", MSE[1])
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        
    #According to project requirements, loss is based on MSE
    loss = tf.losses.mean_squared_error(labels=labels, predictions=decodeds)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = ProjectOptimizer(
            beta=params["B"], learning_rate=params["R"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
            
            
    # Add evaluation metrics (for EVAL mode)
    #According to project requirements, MSE is used to assess the quality of CAE
    eval_metric_ops = {
        "MSE": tf.metrics.mean_squared_error(
            labels=labels, predictions=decodeds)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

#==================== Load training and eval data ===============================
    file_train0 = np.load("./data/data_autoencoder_train.npz")
    train_data0 = file_train0["x_ae_train"].astype(np.float32) / 255
    train_labels0 = file_train0["x_ae_train"].astype(np.float32) / 255
    
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

    file_train = np.load("./data/data_autoencoder_train.npz")
    train_data = file_train["x_ae_train"].astype(np.float32) / 255
    train_labels = file_train["x_ae_train"].astype(np.float32) / 255
    
#=== Numerate the parameters, including learning_rate and beta for momentums =====
    N_RR = 0
    for RR in [0.001, 0.01, 0.1]:
        N_BB = 0
        for BB in [0, 0.1, 0.2, 0.5, 1]:
            print("============== parameters ===============================" +
                  " R=%.4f" % RR + " B=%.4f" % BB)
            RR_BB_accuray = 0
            for K in range(5):
#========= Select one fold for evaluation while others for training =============
                if (K == 0):
                    train_data = np.concatenate(train_data_array[1:5], axis=0)
                    train_labels = np.concatenate(
                        train_labels_array[1:5], axis=0)
                if (K == 4):
                    train_data = np.concatenate(
                        train_data_array[0:4], axis=0).astype(np.float32)
                    train_labels = np.concatenate(
                        train_labels_array[0:4], axis=0)
                if (K < 4 and K > 0):
                    train_data = np.concatenate(
                        np.concatenate(
                            (train_data_array[0:K], train_data_array[K + 1:5]),
                            axis=0),
                        axis=0)
                    train_labels = np.concatenate(
                        np.concatenate(
                            (train_labels_array[0:K],
                             train_labels_array[K + 1:5]),
                            axis=0),
                        axis=0)

                eval_data = train_data_array[K]
                eval_labels = train_labels_array[K]
#=========================  Create the estimator ================================
                # Create the Estimator
                prj_classifier = tf.estimator.Estimator(
                    model_fn=cnn_model_fn,
                    params={
                        "R": RR,
                        "B": BB
                    },
                    model_dir="./trained_model" + "%d" % K + "%d" % N_RR +
                    "%d" % N_BB,
                    config=tf.contrib.learn.RunConfig(num_cores=8))
#======  Check accuracy and loss for every 56000 training samples ================
                evaluation_interval = 56000
                epoch_num = 4
                batch_size_user = 20
                steps_num = evaluation_interval / batch_size_user
                noise_factor = 0.5         # noise_factor is used to insert noise to input images of CAE
                 
                # Set up logging for predictions
                # Log the values in the "Softmax" tensor with label "probabilities"
                tensors_to_log = {"MSE_training": 'MSE_training_training'}
                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=steps_num)
                tf.logging.set_verbosity(tf.logging.INFO)

                for epoch in range(epoch_num):
#========================   Data Shuffling   ============================================
                    s = np.arange(train_data.shape[0])
                    np.random.shuffle(s)
                    train_data = train_data[s]
                    train_labels = train_labels[s]
#========================   Insert Noise to Samples   ===================================
                    noisy_imgs = train_data + noise_factor * np.random.randn(
                        *train_data.shape)
                    # Clip the images to be between 0 and 1
                    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    noisy_imgs = noisy_imgs.astype(np.float32)
                    print("===============" + " R=%.4f" % RR + " B=%.4f" % BB +
                          " Ki=%d" % K + " epoch=%d" % epoch +
                          " =====================")
                    for i in range(
                            int(np.shape(train_data)[0] /
                                evaluation_interval)):
#========================   Execution of Training   =====================================
                        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                            x={
                                "x":
                                noisy_imgs[(i * evaluation_interval):((
                                    i + 1) * evaluation_interval)]
                            },
                            y=train_labels[(i * evaluation_interval):((
                                i + 1) * evaluation_interval)],
                            batch_size=batch_size_user,
                            num_epochs=None,
                            shuffle=True)
                        prj_classifier.train(
                            input_fn=train_input_fn,
                            steps=steps_num,
                            hooks=[logging_hook])

                        # Evaluate the model and print results
                        noisy_imgs_eval = eval_data + noise_factor * np.random.randn(
                            *eval_data.shape)
                        noisy_imgs_eval = np.clip(noisy_imgs_eval, 0., 1.)
                        noisy_imgs_eval = noisy_imgs_eval.astype(np.float32)
                        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                            x={"x": noisy_imgs_eval},
                            y=eval_labels,
                            num_epochs=1,
                            shuffle=False)
                        eval_results = prj_classifier.evaluate(
                            input_fn=eval_input_fn)
                        print(eval_results)
                RR_BB_accuray = RR_BB_accuray + eval_results["MSE"]
                shutil.rmtree(
                    "./trained_model" + "%d" % K + "%d" % N_RR + "%d" % N_BB,
                    True)
            RR_BB_accuray = RR_BB_accuray / 5
            print("============== accuracy == for =======" + " R=%.4f" % RR +
                  " B=%.4f" % BB + "is %.4f" % RR_BB_accuray)
            N_BB = N_BB + 1
        N_RR = N_RR + 1


if __name__ == "__main__":
    tf.app.run()
