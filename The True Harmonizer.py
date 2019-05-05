import py_midicsv
import numpy as np
import tensorflow as tf
import os

csv_string = py_midicsv.midi_to_csv(
    "Music\Rap\Four_five_seconds_-_Rihanna_Paul_McCartney_Kanye_West.mid")

information = []
highnotes = []
bassnotes = []

for note in csv_string:
    temp = note.split(',')
    for attribute in temp:
        if attribute == '1':
            for check in temp:
                if check == ' Note_on_c':
                    highnotes.append(temp)
        elif attribute == '2':
            for check in temp:
                if check == ' Note_on_c':
                    bassnotes.append(temp)
        else:
            temp=temp

highnumpynotes = np.array(highnotes)
bassnumpynotes = np.array(bassnotes)
print(highnumpynotes.shape)
print(bassnumpynotes.shape)
print(bassnumpynotes)

LOGDIR = './graphs'

sess = tf.Session()

#Hyper parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 300
EPOCHS = 10

#Hidden Layers
HL_1 = 1000
HL_2 = 500

#Other parameters
INPUT_SIZE = 336 * 6
N_CLASSES = 940*6

with tf.name_scope('input'):
    inputnotes = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="inputnotes")
    labels = tf.placeholder(tf.float32, [None, N_CLASSES], name="outputnotes")
    
def fc_layer(x, layer, size_out, activation=None):
    with tf.name_scope('layer'):
        size_in = int(x.shape[1])
        W = tf.Variable(tf.random_normal([size_in, size_out]), name="weights")
        b = tf.Variable(tf.Constant(-1, dtype=tf.float32, shape=[size_out]), name="biases")
        
        wx_plus_b = tf.add(tf.matmul(x, W), b)
        if activation:
            return activation(wx_plus_b)
        return wx_plus_b

fc_1 = fc_layer(inputnotes, 'fc_1', HL_1, tf.nn.relu)
fc_2 = fc_layer(fc_1, 'fc_2', HL_2, tf.nn.relu)

#to prevent overfitting
dropped = tf.nn.dropout(fc_2, keep_prob=.9)

y=fc_layer(dropped, 'output', N_CLASSES)



with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.name_scope('evaluation'):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"), sess.graph)

summary_op = tf.summary.mearge_all()

init = tf.global_variables_initializer()
sess.run(init)

with tf.name_scope('training'):
    step = 0
    for epoch in range(EPOCHS):
        print("epoch", epoch, "\n--------------------\n")

        for batch in range(int(highnumpynotes.train.labels.shape[0] / BATCH_SIZE)):
            step += 1
            batch_xs = highnumpynotes.train.next_batch(BATCH_SIZE)
            batch_ys = bassnumpynotes.train.next_batch(BATCH_SIZE)

            summary_result, _ = sess.run([summary_op, train], feed_dict={inputnotes: batch_xs, labels: batch_ys})

            train_writer.add_summary(summary_result, step)

            summary_result, acc = sess.run([summary_op, accuracy], feed_dict={inputnotes: highnumpynotes.test.inputnotes, labels: bassnumpynotes.test.labels})
            
            test_writer.add_summary(summary_result, step)

            print("Batch ", batch, ": accuracy = ", acc)

train_writer.close()
test_writer.close()
sess.close()
            