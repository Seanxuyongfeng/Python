import tensorflow as tf

a = tf.placeholder(tf.float32, name="a")
b = tf.placeholder(tf.float32, name="b")
c = tf.multiply(a,b, name="c")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    #无需初始化，因为没有用到variable
    #sess.run(init)
    
    result = sess.run(c, feed_dict={a:8.0,b:9.0})
    
    print(result)