import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
   

X_train, y_train = train['features'], train['labels']

n_train = len(X_train)
image_shape = X_train[0].shape


# normalise the images
pixel_depth = 255.0
NORMALISE_BATCH_SIZE = 5000

X_train_normalised = np.ndarray(shape=(n_train, image_shape[0], image_shape[1],3),
                         dtype=np.float32)

for offset in range(0, n_train, NORMALISE_BATCH_SIZE):
    batch_end = offset+NORMALISE_BATCH_SIZE
    X_train_normalised[offset:batch_end] = (X_train[offset:batch_end] - pixel_depth / 2) / pixel_depth

train_features, valid_features, train_labels, valid_labels = train_test_split(
    X_train_normalised,
    y_train,
    test_size=0.2,
    random_state=832289)

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# TODO: Define placeholders and resize operation.
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)


EPOCHS = 2
BATCH_SIZE = 128


# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

fc8W = tf.Variable(tf.truncated_normal(shape, mean = 0, stddev = 0.01))
fc8b = tf.Variable(tf.zeros(shape[1]))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_features)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_, y_train_ = shuffle(train_features, train_labels)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = min(offset + BATCH_SIZE,num_examples-1)
            batch_x, batch_y = X_train_[offset:end], y_train_[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x,y: batch_y})

        validation_accuracy = evaluate(valid_features, valid_labels)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'traffic_sign_classifier')
    print("Model saved")




# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))

#     test_accuracy = evaluate(X_test_grayscale, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))