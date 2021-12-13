import numpy as np
import tensorflow as tf

'''
# 梯度下降例子
# 构造数据集
x_pure = np.random.randint(-10, 100, 32)
x_train = x_pure + np.random.randn(32) / 32
y_train = 3 * x_pure + 2 + np.random.randn(32) / 32

x_input = tf.placeholder(tf.float32, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input')
w = tf.Variable(2.0, name='weight')
b = tf.Variable(1.0, name='biases')
y = tf.add(tf.multiply(x_input, w), b)

loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)
gradients_node = tf.gradients(loss_op, w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20):
    _, gradients, loss = sess.run([train_op, gradients_node, loss_op], feed_dict={x_input: x_train[i], y_input: y_train[i]})
    print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))
sess.close()
'''

#### Stop_gradient的使用
"""
a = tf.constant(0.)
b = 2 * a
c = a + b
g = tf.gradients(c, [a, b])

# test gradient calc. with session.run()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(g))
"""

# test graph defination without session.run()
"""
before:
-w1->(*3.0)-a->(*w2)-b->
after stop gradient:
-w1->(*3.0)-a->(stop_grad_node)-a_stoped->(*w2)-b->
"""
from tensorflow.contrib import graph_editor as ge
w1 = tf.Variable(2.0)
w2 = tf.Variable(2.0)
a = tf.multiply(w1, 3.0)
a_stoped = tf.stop_gradient(a)
# b=w1*3.0*w2
b = tf.multiply(a_stoped, w2)
gradients = tf.gradients(b, xs=[w1, w2])
print(ge.get_tensors(tf.get_default_graph()))
print(gradients)
# 输出[None, <tf.Tensor 'gradients/Mul_1_grad/Reshape_1:0' shape=() dtype=float32>]

# 在图构建后再停止梯度，需要重新连接grad node的边
# 以下操作等价于上段
"""
call stop_gradient after graph building
-w1->(*3.0)-a->(*w2)-b->
            \
             (stop_grad_node)-grad_node->
re-wire edges
-w1->(*3.0)-a->(stop_grad_node)-grad_node->(*w2)-b->
"""

# from tensorflow.contrib import graph_editor as ge
# w1 = tf.Variable(2.0)
# w2 = tf.Variable(2.0)
# a = tf.multiply(w1, 3.0)
# # b=w1*3.0*w2
# b = tf.multiply(a, w2)
# grad_node = tf.stop_gradient(a, name=a.op.name+"_sg")
# ge.reroute_ts([grad_node], [a])
# gradients = tf.gradients(b, xs=[w1, w2])
# print(ge.get_tensors(tf.get_default_graph()))
# print(gradients)
# # 输出[None, <tf.Tensor 'gradients/Mul_1_grad/Reshape_1:0' shape=() dtype=float32>]



# has another branch, thus [a,b] still have gradient
"""
a = tf.Variable(1.0)
b = tf.Variable(1.0)
c = tf.add(a, b)
c_stoped = tf.stop_gradient(c)
d = tf.add(a, b)
e = tf.add(c_stoped, d)
gradients = tf.gradients(e, xs=[a, b])
print(gradients)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(gradients))
    #因为梯度从另外地方传回，所以输出 [1.0, 1.0]
"""