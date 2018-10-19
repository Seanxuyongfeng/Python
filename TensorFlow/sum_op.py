import tensorflow as tf

value = tf.Variable(0)
sumvalue = tf.Variable(0)
one = tf.constant(1)

next_value = tf.add(value, one)
update_value = tf.assign(value, next_value)

sum_op = tf.add(sumvalue, update_value)
assign_sum = tf.assign(sumvalue, sum_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(assign_sum)
        print(sess.run(value))
    print(sess.run(sumvalue))