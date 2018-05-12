import tensorflow as tf
# labels = tf.Variable([[4, 3, 1, 2, 5],
#                       [2, 3, 4, 1, 0],
#                       [1, 2, 3, 0, 0],
#                       [5, 4, 0, 0, 0]], tf.int32)
# print labels.shape
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     idx = tf.where(tf.not_equal(labels, 0))
#     sparse = tf.SparseTensor(idx, tf.gather_nd(
#         labels, idx), labels.get_shape())
#     s = sess.run(sparse)
#     print s.indices
#     print s.values
#     print s.dense_shape

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
