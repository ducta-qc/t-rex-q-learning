import tensorflow as tf
import threading


def create_queue_ops(
        batch_size, 
        frame_width=80, 
        frame_height=80, 
        frame_channels=4,  
        capacity=1024):
    frame_input = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
    action_input = tf.placeholder(tf.int32, shape=[None, 1])
    reward_input = tf.placeholder(tf.float32, shape=[None, 1])

    queue = tf.FIFOQueue(
                capacity=capacity, 
                dtype=[tf.float32, tf.int32, tf.float32], 
                shapes=[[frame_width, frame_height, frame_channels], [1], [1]])

    enqueue_op = queue.enqueue_many([frame_input, action_input, reward_input])
    dequeue_op = queue.dequeue()
    data_batch = tf.train.batch([dequeue_op], 
                    batch_size=batch_size, capacity=capacity, num_threads=3)

    return data_batch, enqueue_op, dequeue_op, (frame_input, action_input, reward_input)


def run_enqueue(sess, prepared_queue, enqueue_op, inputs):
    while(True):
        while not prepared_queue.empty():
            pass

        item = prepared_queue.get()
        lfr = [k[0] for k in item]
        lac = [k[1] for k in item]
        lrw = [k[2] for k in item]

        frame = np.stack([lfr], dtype=np.float32)
        action = np.asarray(lac[-1])    
        reward = np.asarray(lrw[-1])

        try:
            sess.run(enqueue_op, feed_dict={inputs[0]:frame, inputs[1]:action, inputs[2]:reward})
        except:
            break

def t_rex_game_learning():
    data_batch, enqueue_op, dequeue_op, inputs = create_queue_ops(512)
    train_dir = '/tmp/t-rex/train'
    cpkt_dir = '/tmp/t-rex/checkpoint'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(cpkt_dir):
        os.makedirs(cpkt_dir)


    training_saver = tf.train.Saver(var_list=None, max_to_keep=10)

    pass