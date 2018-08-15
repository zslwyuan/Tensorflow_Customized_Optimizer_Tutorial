from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import matplotlib.pyplot as plt
import argparse

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
        m_t = m.assign(
            beta_t * m + lr_t * grad)  # update the gradients with momentums
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
        "Decoded": decodeds,
        "Features_map": encoded
    }
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
    #According to project requirements, MSE is used to assess the quality of CAE
    MSE = tf.metrics.mean_squared_error(
        labels=labels, predictions=decodeds)  
        
    #Use tf.summary.scalar to log data for tensorboard figure generation
    tf.identity(MSE[1], name="MSE_training_training")
    tf.summary.scalar("MSE_training", MSE[1])
    
    #Calculate Loss (for both TRAIN and EVAL modes), a Tensor of loss
    loss = tf.losses.mean_squared_error(labels=labels, predictions=decodeds)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = ProjectOptimizer(
            beta=params["beta"], learning_rate=params["learning_rate"])
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
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        predictions=decodeds)

#================ Execution Flow of Training and Evaluation ===============================
#refer to https://www.tensorflow.org/get_started/custom_estimators
def main(unused_argv):

    tf.set_random_seed(1234)
    np.random.seed(1234)
#==================== argument controlling the output of figures ==================
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--nofigure',  action="store_true",default=False,
                        help='Do not present figures during training')
    args = parser.parse_args()
    
    Show_figures = not args.nofigure
    
#==================== Load training and eval data ===============================
    file_train = np.load("./data/data_autoencoder_train.npz")
    train_data = file_train["x_ae_train"].astype(np.float32) / 255
    train_labels = file_train["x_ae_train"].astype(np.float32) / 255
    file_test = np.load("./data/data_autoencoder_eval.npz")
    eval_data = file_test["x_ae_eval"].astype(np.float32) / 255
    eval_labels = file_test["x_ae_eval"].astype(np.float32) / 255
    
#=========================  Create the estimator ================================
    prj_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={
            "learning_rate": 0.1,
            "beta": 0.5
        },
        model_dir="./trained_model_CAE",
        config=tf.contrib.learn.RunConfig(num_cores=8))

    evaluation_interval = 1000 # Check the evaluation accuracy per 1000 samples of training
    epoch_num = 4
    batch_size_user = 20
    steps_num = evaluation_interval / batch_size_user
    noise_factor = 0.5         # noise_factor is used to insert noise to input images of CAE 
    
    # Set up logging for tensorboard figures
    # Log the values in the "MSE_training_training" tensor with label "MSE_training" for tensor
    tensors_to_log = {"MSE_training": 'MSE_training_training'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=steps_num)

#========================   Initialization  ===========================================
    if (Show_figures):
        plt.ion()
        fig0, axes0 = plt.subplots(
            nrows=3, ncols=10, sharex=True, sharey=True, figsize=(10, 3))
        fig0.canvas.set_window_title('Figures - Original/Noise-inserted/Denoised') 
        fig1, axes1 = plt.subplots(
            nrows=2, ncols=10, sharex=True, sharey=True, figsize=(10, 2))
        fig1.canvas.set_window_title('The Figure_maps corresponding to Figures in anther window') 
#========================   Major Works Start  ===========================================

    for epoch in range(epoch_num):
        print(
            "=============== epoch" + "%d" % epoch + " =====================")
#========================   Data Shuffling   ============================================
        s = np.arange(train_data.shape[0])
        np.random.shuffle(s)
        train_data = train_data[s]
        train_labels = train_labels[s]
#========================   Insert Noise to Samples   ===================================        
        noisy_imgs = train_data + noise_factor * np.random.randn(
            *train_data.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        noisy_imgs = noisy_imgs.astype(np.float32)
        
        
        for i in range(int(np.shape(train_data)[0] / evaluation_interval)):
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
                input_fn=train_input_fn, steps=steps_num, hooks=[logging_hook])
                
                
#=====================Periodical Evaluation============================================= 
            noisy_imgs_eval = eval_data + noise_factor * np.random.randn(
                *eval_data.shape)
            noisy_imgs_eval = np.clip(noisy_imgs_eval, 0., 1.)
            noisy_imgs_eval = noisy_imgs_eval.astype(np.float32)
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": noisy_imgs_eval},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = prj_classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)
            
#===============Reconstruct some figures with noise inserted and plot them ===============
            noisy_imgs_eval = eval_data[:10] + noise_factor * np.random.randn(
                *eval_data[:10].shape)
            # Clip the images to be between 0 and 1
            noisy_imgs_eval = np.clip(noisy_imgs_eval, 0., 1.)
            noisy_imgs_eval = noisy_imgs_eval.astype(np.float32)
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": noisy_imgs_eval},
                y=eval_labels[:10],
                num_epochs=1,
                shuffle=False)
            predict_result = prj_classifier.predict(input_fn=eval_input_fn)
            tj = 0
            reconstructed0 = []
            reconstructed1 = []
            TMP0_reconstructed1=[]
            TMP1_reconstructed1=[]
            for p in predict_result:
                b = np.clip(p['Decoded'], 0., 1.)
                b = b * 255
                reconstructed0.append(b.astype(int))
                b = np.clip(p['Features_map'], 0., 1.)
                b = b * 255
                TMP0_reconstructed1.append(b.astype(int)[:,:,0])
                TMP1_reconstructed1.append(b.astype(int)[:,:,1])
               # reconstructed1.append(b.astype(int))
                tj = tj + 1
                if (tj >= 10):
                    break
            
#========================= Plot the reconstructed figures ================================
            if (Show_figures):
                for images, row in zip(
                    [eval_data[:10], noisy_imgs_eval, reconstructed0], axes0):
                    for img, ax in zip(images, row):
                        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                fig0.tight_layout(pad=0.1)
                
                for images, row in zip(
                    [TMP0_reconstructed1, TMP1_reconstructed1], axes1):
                    for img, ax in zip(images, row):
                        ax.imshow(img.reshape((7, 7)), cmap='Greys_r')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                fig1.tight_layout(pad=0.1)
                
                plt.draw()
                plt.pause(2)


if __name__ == "__main__":
    tf.app.run()
