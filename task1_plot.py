from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

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
        m_t = m.assign(beta_t * m + lr_t * grad) # update the gradients with momentums
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

    logits = tf.layers.dense(inputs=dropout, units=47)

    predictions = { # Generate predictions (for PREDICT and EVAL mode)
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    #According to project requirements, logstic is used to assess the quality of CNN

        
        
#======== Compute the accuracy and loss of the CNN model on the training dataset =========
    #Use tf.summary.scalar to log data for tensorboard figure generation

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar("prediction_accuracy", accuracy[1])
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = ProjectOptimizer(
            beta=params["beta"], learning_rate=params["learning_rate"])
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
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,predictions=predictions)


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
    file_train = np.load("./data/data_classifier_train.npz")
    train_data = file_train["x_train"].astype(np.float32)/255
    train_labels = file_train["y_train"].astype(np.int32)
    file_test = np.load("./data/data_classifier_test.npz")
    eval_data = file_test["x_test"].astype(np.float32)/255
    eval_labels = file_test["y_test"].astype(np.int32)
    
#=========================  Create the estimator ================================
    prj_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={
            "learning_rate": 0.01,
            "beta": 0.5
        },
        model_dir="./trained_model_CNN",
        config=tf.contrib.learn.RunConfig(num_cores=8))
        
#========================   Initialization  ===========================================
    evaluation_interval = 1000
    epoch_num = 4
    batch_size_user = 50
    steps_num = evaluation_interval / batch_size_user

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # Log the values in the "train_accuracy" tensor with label "prediction_accuracy"
    tensors_to_log = {
        "probabilities": "softmax_tensor",
        "prediction_accuracy": "train_accuracy"
    }
#========================   Initialization of figures ===================================
    if (Show_figures):
        plt.ion()
        fig0, axes0 = plt.subplots(
            nrows=10, ncols=10, sharex=True, sharey=True, figsize=(10, 10))
        fig0.canvas.set_window_title('Figures classied into 1,3,5,7,9,11...') 
        
#========================   Major Works Start  ===========================================
    for epoch in range(epoch_num):
        print(
            "=============== epoch" + "%d" % epoch + " =====================")
#========================   Data Shuffling   ============================================
        s = np.arange(train_data.shape[0])
        np.random.shuffle(s)
        train_data = train_data[s]
        train_labels = train_labels[s]
        for i in range(int(np.shape(train_data)[0] / evaluation_interval)):
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
            prj_classifier.train(
                input_fn=train_input_fn, steps=steps_num)

#=====================Periodical Evaluation=============================================
#===== Compute the accuracy and loss of the CNN model on the evaluation dataset ========
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
            eval_results = prj_classifier.evaluate(input_fn=eval_input_fn)
            print("Accuracy and loss for Evaluation dataset")
            print(eval_results)
            
#========================= Plot the reconstructed figures ================================
            predict_result = prj_classifier.predict(input_fn=eval_input_fn)
            tj = 0
            amo = 0
            
            a1 = []           
            a3 = []
            a5 = []
            a7 = []
            a9 = []
            
            a11 = []            
            a13 = []
            a15 = []            
            a17 = []
            a19 = []
            
            for p in predict_result:
                b = p["classes"]
                if (b==1 and len(a1)<10):
                    a1.append(eval_data[tj])
                    amo = amo + 1
                if (b==3 and len(a3)<10):
                    a3.append(eval_data[tj])
                    amo = amo + 1                
                if (b==5 and len(a5)<10):
                    a5.append(eval_data[tj])
                    amo = amo + 1
                if (b==7 and len(a7)<10):
                    a7.append(eval_data[tj])
                    amo = amo + 1
                if (b==9 and len(a9)<10):
                    a9.append(eval_data[tj])
                    amo = amo + 1
                if (b==11 and len(a11)<10):
                    a11.append(eval_data[tj])
                    amo = amo + 1
                if (b==13 and len(a13)<10):
                    a13.append(eval_data[tj])
                    amo = amo + 1             
                if (b==15 and len(a15)<10):
                    a15.append(eval_data[tj])
                    amo = amo + 1     
                if (b==17 and len(a17)<10):
                    a17.append(eval_data[tj])
                    amo = amo + 1                     
                if (b==19 and len(a19)<10):
                    a19.append(eval_data[tj])
                    amo = amo + 1                     
                if (amo>=100):
                    break
                tj = tj + 1
                    
            if (Show_figures):
                for images, row in zip(
                    [a1,a3,a5,a7,a9,a11,a13,a15,a17,a19], axes0):
                    for img, ax in zip(images, row):
                        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                fig0.tight_layout(pad=0.1)
                
                plt.draw()
                plt.pause(2)
            
#=====================Final Evaluation for training set =================================

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, num_epochs=1, shuffle=False)
    eval_results = prj_classifier.evaluate(input_fn=eval_input_fn)
    print("Accuracy and loss for Training dataset")
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
