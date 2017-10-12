import tensorflow as tf
import threading
from model import TRexCNN
import time
import numpy as np
import os


class TRexGaming(object):
    def __init__(self, prepared_queue, batch_size=512, using_cuda=True):
        self.prepared_queue = prepared_queue
        self.batch_size = batch_size
        self.using_cuda = using_cuda
        self.model = TRexCNN(using_cuda=using_cuda)
        self.dev = "/cpu:0"
        if using_cuda:
            self.dev = "/gpu:0"

    def create_input_tensors(self, batch_size=512, for_training=True):
        curr_state_input = tf.placeholder(tf.float32, shape=[batch_size, 80, 80, 4])
        if not for_training:
            return curr_state_input
        action_input = tf.placeholder(tf.int32, shape=[batch_size])
        reward_input = tf.placeholder(tf.float32, shape=[batch_size])
        next_state_input = tf.placeholder(tf.float32, shape=[batch_size, 80, 80, 4])
        return curr_state_input, action_input, reward_input, next_state_input

    @staticmethod
    def train_thread(sess, train_op, loss, inputs,
                     prepared_queue, batch_size, max_steps,
                     save_steps, training_saver, dev):
        step = 0
        with tf.device(dev):
            while(True):
                if step==0 : time.sleep(1)
                i = 0
                curr_state = []
                action = []
                reward = []
                next_state = []
                while i < batch_size:
                    item = prepared_queue.get()
                    if item == None:
                        continue
                    curr_state.append(item[0])
                    action.append(item[1])
                    reward.append(item[2])
                    next_state.append(item[3])
                    i+=1
                curr_state = np.stack(curr_state, axis=0)
                action = np.stack(action, axis=0)
                reward = np.stack(reward, axis=0)
                next_state = np.stack(next_state, axis=0)
                _, ret_loss = sess.run([train_op, loss],
                                feed_dict={inputs[0]:curr_state, inputs[1]:action, inputs[2]:reward, inputs[3]:next_state})
                #train_op.run(feed_dict={inputs[0]:curr_state, inputs[1]:action, inputs[2]:reward, inputs[3]:next_state},
                #             session=sess)
                step += 1
                #if step % 2 == 0:
                print("Step %d: %0.3f" % (step, ret_loss))
                if step >= max_steps:
                    break
            sess.close()

    @staticmethod
    def train_thread_t():
        while(True):
            time.sleep(1)
    def learning(self, max_steps=20000, save_steps=500):
        train_dir = '/tmp/t-rex/train'
        cpkt_dir = '/tmp/t-rex/checkpoint'

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(cpkt_dir):
            os.makedirs(cpkt_dir)

        with tf.Graph().as_default(), tf.device("/cpu:0"):
            # Build tensorflow ops for predicting and training processes
            inputs = self.create_input_tensors()
            self.curr_state_input = self.create_input_tensors(batch_size=1, for_training=False)
            self.pred_q_value, self.pred_action = \
                self.model.build_predict_op([self.curr_state_input, 0.2])
            train_op, global_step, loss = self.model.build_train_op(inputs)

            training_saver = tf.train.Saver(var_list=None, max_to_keep=10)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            #TODO restore pre-trained session
            # Start training thread
            train_thread = threading.Thread(
                target=TRexGaming.train_thread,
                args=[self.sess, train_op, loss, inputs, self.prepared_queue,
                      self.batch_size, max_steps, save_steps, training_saver, self.dev])
            # train_thread = threading.Thread(target=TRexGaming.train_thread_t, args=[])
            train_thread.daemon = True
            train_thread.start()

    def take_a_action(self, curr_state):
        with tf.device(self.dev):
            q_value, action = self.sess.run(
                                    [self.pred_q_value, self.pred_action],
                                    feed_dict={self.curr_state_input:curr_state})
        return action

    def restart(self):
        pass