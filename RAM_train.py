import tensorflow as tf
from RAM_model import *
from utils import *
import tf_mnist_loader
import time

def evaluate(dataset):
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0
        
    for i in xrange(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        feed_dict = {img: nextX, labels: nextY, tensor_labels: dense_to_one_hot(nextY)}          
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))  

dataset = tf_mnist_loader.read_data_sets("mnist_data")

optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(cost)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

if load_path is not None:
    try:
        saver.restore(sess, load_path)
        print("LOADED SAVED MODEL")
    except:
        print("FAILED TO LOAD SAVED MODEL")
        exit()
else:
    init = tf.initialize_all_variables()
    sess.run(init)


for step in xrange(max_iters):
    start_time = time.time()
    
    nextX, nextY = dataset.train.next_batch(batch_size)
    feed_dict = {img: nextX, labels: nextY, tensor_labels: dense_to_one_hot(nextY)}
    fetches = [train_op, cost, reward]
    
    results = sess.run(fetches, feed_dict=feed_dict)
    _, cost_fetched, reward_fetched = results
    
    duration = time.time() - start_time

    if (step + 1) % 20 == 0:
        print('Step %d: cost = %.5f reward = %.5f (%.3f sec)' % (step + 1, cost_fetched, reward_fetched, duration))

    if (step + 1) % 2000 == 0:
        evaluate(dataset) 

    if (step + 1) % 4000 == 0:
        saver.save(sess, save_dir + save_prefix + ".ckpt")
        print("Model Saved")
    
     










