import tensorflow as tf

dossier = "train_data/*/*.png"
filenames = tf.train.match_filenames_once(dossier)
queue = tf.train.string_input_producer(filenames)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    
    sess.run(init)
    print(sess.run(filenames))  # GOOD
    print(sess.run(queue.size())) 
    