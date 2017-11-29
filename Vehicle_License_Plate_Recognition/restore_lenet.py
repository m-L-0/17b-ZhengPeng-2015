import tensorflow as tf


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./my_model/.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./my_model/checkpoint'))
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)
