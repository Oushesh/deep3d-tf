import tensorflow as tf
import Deep3D_Final as deep3d
import utils
import numpy as np
import os
import os.path
import h5py
import matplotlib.pyplot as plt

batchsize = 50
num_epochs = 20
print_step = 1


left_dir = "frames/train/left/"
right_dir = "frames/train/right/"

with tf.device('/gpu:0'):
    left_image_queue = tf.train.string_input_producer(
      left_dir + tf.convert_to_tensor(os.listdir(left_dir)),
      shuffle=False, num_epochs=num_epochs)
    right_image_queue = tf.train.string_input_producer(
      right_dir + tf.convert_to_tensor(os.listdir(right_dir)),
      shuffle=False, num_epochs=num_epochs)

    # use reader to read file
    image_reader = tf.WholeFileReader()

    _, left_image_raw = image_reader.read(left_image_queue)
    left_image = tf.image.decode_jpeg(left_image_raw)
    left_image = tf.cast(left_image, tf.float32)/255.0

    _, right_image_raw = image_reader.read(right_image_queue)
    right_image = tf.image.decode_jpeg(right_image_raw)
    right_image = tf.cast(right_image, tf.float32)/255.0

    left_image.set_shape([160,288,3])
    right_image.set_shape([160,288,3])

    # preprocess image
    batch = tf.train.shuffle_batch([left_image, right_image],
                                   batch_size = batchsize,
                                   capacity = 12*batchsize,
                                   num_threads = 1,
                                   min_after_dequeue = 4*batchsize)


# Define config for GPU memory debugging
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  # Switch to True for dynamic memory allocation instead of TF hogging BS
config.gpu_options.per_process_gpu_memory_fraction= 1  # Cap TF mem usage
config.allow_soft_placement=True


# Session
sess = tf.Session(config=config)

# Placeholders
images = tf.placeholder(tf.float32, [None, 160, 288, 3], name='input_batch')
true_out = tf.placeholder(tf.float32, [None, 160, 288, 3] , name='ground_truth')
train_mode = tf.placeholder(tf.bool, name='train_mode')

# Building Net based on VGG weights
net = deep3d.Deep3Dnet('./vgg19.npy', dropout = 1.0)
net.build(images, train_mode)

# Print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print ('Variable count:')
print(net.get_var_count())

# Define Training Objectives
with tf.variable_scope("Loss"):
    cost = tf.reduce_sum(tf.abs(net.prob - true_out))/batchsize

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(cost)

# Run initializer
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
queue_threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Track Cost
tf.summary.scalar('cost', cost)
# tensorboard operations to compile summary and then write into logs
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tensorboard_logs/', graph = sess.graph)


# Training Loop
print ("")
print ("== Start training ==")

#base case
next_batch = sess.run(batch)
i=0
try:
    while not coord.should_stop():

            # Traing Step
        _, cost_val, next_batch, summary, up_conv = sess.run([train, cost, batch, merged, net.up_conv],
                                                    feed_dict={images: next_batch[0],
                                                               true_out: next_batch[1],
                                                               train_mode: True})
        writer.add_summary(summary, i)

        # No longer needed: cost_hist.append(cost_val)
        if i%print_step == 0:
            print (str(i) + ' | Cost: ' + str(cost_val) + " | UpConv Max: " + str(np.mean(up_conv, axis =(0,1,2)).max()))
        i+=1

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

#finally:
    # When done, ask the threads to stop.
    #coord.request_stop()


print ("")
print ("Training Completed, storing weights")
# Store Traing Output
net.save_npy(sess)


#termination stuff
coord.request_stop()
coord.join(queue_threads)
sess.close()

#Testing Output
inria_file = 'deep3d_data/inria_data.h5'
# inria_file = 'data/inria_data.h5'
h5f = h5py.File(inria_file,'r')
# X_train_0 = h5f['X_0'][:,10:170,16:304,:]
# Y_train_0 = h5f['Y_0'][:,10:170,16:304,:]
# X_train_1 = h5f['X_1'][:,10:170,16:304,:]
# Y_train_1 = h5f['Y_1'][:,10:170,16:304,:]
# X_train_2 = h5f['X_2'][:,10:170,16:304,:]
# Y_train_2 = h5f['Y_2'][:,10:170,16:304,:]
# X_train_3 = h5f['X_3'][:,10:170,16:304,:]
# Y_train_3 = h5f['Y_3'][:,10:170,16:304,:]
# X_train_4 = h5f['X_4'][:,10:170,16:304,:]
# Y_train_4 = h5f['Y_4'][:,10:170,16:304,:]
# X_train_5 = h5f['X_5'][:,10:170,16:304,:]
# Y_train_5 = h5f['Y_5'][:,10:170,16:304,:]
# X_train_6 = h5f['X_6'][:,10:170,16:304,:]
# Y_train_6 = h5f['Y_6'][:,10:170,16:304,:]
# #X_train_7 = h5f['X_7'][:,10:170,16:304,:]
# #Y_train_7 = h5f['Y_7'][:,10:170,16:304,:]

X_val = h5f['X_7'][:,10:170,16:304,:]
Y_val = h5f['Y_7'][:,10:170,16:304,:]

h5f.close()

# # ------------------------------------------#
# X_train = np.concatenate([X_train_0,X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6])
# Y_train = np.concatenate([Y_train_0,Y_train_1,Y_train_2,Y_train_3,Y_train_4,Y_train_5,Y_train_6])

# print "Training Size:" + str(X_train.shape)
print ("Validation Size:" + str(X_val.shape))

# Test
test_img = np.expand_dims(X_val[365], axis = 0)
test_ans = Y_val[365]

with tf.device("/gpu:0"):
    res, mask, up_conv = sess.run([net.prob, net.mask, net.up_conv],
                                  feed_dict={images: test_img, train_mode: False})

print ("--- Input ---")
plt.imshow(test_img[0])
plt.show()

print ("--- GT ---")
plt.imshow(test_ans)
plt.show()

print ("--- Our result ---")
plt.imshow(res[0])
plt.show()
