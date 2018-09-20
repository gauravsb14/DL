import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
n_input = mnist.train.images.shape[1]
n_hidden = 200
n_classes = 10

def MLP_graph():
    x = tf.placeholder("float",[None,n_input])
    y  =  tf.placeholder("float",[None, n_classes])
    weights = {
        "h1": tf.Variable(tf.random_normal([n_input,n_hidden])),
        "out" : tf.Variable(tf.random_normal([n_hidden,n_classes]))
    }
    return x,y,weights

def MLP_model(x,y, weights):
    h_input = tf.nn.relu(tf.matmul(x,weights["h1"]))
    logits = tf.matmul(h_input, weights["out"])
    pred = tf.one_hot(tf.cast(tf.argmax(logits,1),tf.int32), depth = 10)

    return pred, logits

def get_loss(logits,y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits= logits))
    return loss

def get_accuracy(pred, y):
    correct_predction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predction,tf.float32))
    return accuracy

def main():
    x, y, weights = MLP_graph()
    pred, logits = MLP_model(x,y, weights)
    loss = get_loss(logits,y)
    accuracy = get_accuracy(pred, y)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch = mnist.train.next_batch(1000)
        # print(batch[1])
        if (i % 100) == 0:
            acc  = sess.run(accuracy, feed_dict={x:batch[0],y:batch[1]})
            print("test accuracy at step %s: %s" % (i,acc))
        else:
            sess.run(train_step,feed_dict={x:batch[0],y:batch[1]})
    print("Accuracy using tensorflow is :")
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

main()

