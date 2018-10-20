
# coding: utf-8

# In[28]:

import tensorflow as tf

retry = True

while retry:
    n = input('Input n: ')
    try:
        n = int(n)
        retry = False
    except:
        print("Input Error,Please try again!")
        retry = True
        continue
    finally:
        pass
value = tf.Variable(0)
sumvalue = tf.Variable(0)
one = tf.constant(1)

input_n = tf.placeholder(tf.int32, name="input_n")

next_value = tf.add(value, one)
update_value = tf.assign(value, next_value)

sum_op = tf.add(sumvalue, update_value)
assign_sum = tf.assign(sumvalue, sum_op)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    count = sess.run(input_n,feed_dict={input_n:n})
    for i in range(count):
        sess.run(assign_sum)
        print(sess.run(value))
    print(sess.run(sumvalue))

