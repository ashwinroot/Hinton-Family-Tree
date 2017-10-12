import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


error_matrix = []
iter_matrix = []
train_matrix = []
test_matrix = []

def familytree():
    def bitvec(ix,nbit):
        out = []
        for i in range(nbit):
            out.append((i==ix)+0.0)
        return np.array(out)

    names = [ "Christopher", "Andrew", "Arthur", "James", "Charles", "Colin", "Penelope", "Christine", "Victoria", "Jennifer", "Margaret", "Charlotte", "Roberto", "Pierro", "Emilio", "Marco", "Tomaso", "Alfonso", "Maria", "Francesca", "Lucia", "Angela", "Gina", "Sophia"]
    relations = [ "husband", "wife", "son", "daughter", "father", "mother", "brother", "sister", "nephew", "niece", "uncle", "aunt"]

    dataset = []
    with open('/Users/ashwinsankar/Desktop/DL/HW 3/relations.txt','r') as f:
        for line in f:
            sline = line.split();
            p1 = names.index(sline[0])
            r = relations.index(sline[1])
            p2 = names.index(sline[2])
            d = [ sline[0]+'-'+sline[1]+'-'+sline[2],
                  np.concatenate((bitvec(p1,len(names)),bitvec(r,len(relations)))),
                  bitvec(p2,len(names)) ]
                  #bitvec(p2,len(names)) ]
            dataset.append(d)

    return dataset

def model_build(n_dim,n_hidden,n_class):
    #hidden layer params
    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([n_dim,n_hidden])),
                     'bias' : tf.Variable(tf.random_normal([n_hidden]))}
    #output layer params
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden, n_class])),
                     'bias': tf.Variable(tf.random_normal([n_class]))}

    #construction of connections
    l1 = tf.add(tf.matmul(x,hidden_layer1['weights']) , hidden_layer1['bias'])
    l1 = tf.nn.sigmoid(l1)
    out = tf.matmul(l1,output_layer['weights'])
    return out,hidden_layer1,output_layer


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.array(data_shuffle), np.array(labels_shuffle)

def random_shuffle(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    return data_shuffle

def train(train_x,train_y,test_x,test_y,n_hidden,n_classes,batch_size,n_epoches):
    prediction = model_build(train_x.shape[1],n_hidden,n_classes)
    cost = tf.reduce_sum(tf.square(prediction - y)/2)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    correct = abs(tf.round(prediction) - y )
    # correct = tf.equal(prediction,y)
    train_accuracy = tf.reduce_sum(correct)
    test_accuracy = tf.reduce_sum(abs(correct))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    hm_epochs = n_epoches
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for epoch in range(0,hm_epochs):
            epoch_loss = 0
            train_acc = 0
            i = 0
            for p in range(int(train_x.shape[0]/batch_size)):
                x_,y_ = next_batch(batch_size,train_x,train_y)
                _,c, acc_i = sess.run((optimizer,cost, train_accuracy) , feed_dict={x:x_, y: y_})
                train_acc+=acc_i
                epoch_loss+=c
            train_acc = 100 - train_acc/batch_size
            epoch_loss /= train_x.shape[0]
            test_acc = 100 - sess.run(test_accuracy,feed_dict={x:test_x,y:test_y}) # subtracting 100 from the percentage of error
            print('Epoch:',epoch,'loss:',epoch_loss, 'Train-Accuracy:',train_acc, 'Test-Accuracy:', test_acc )
            test_matrix.append(test_acc)
            train_matrix.append(train_acc)
            error_matrix.append(epoch_loss)
            iter_matrix.append(epoch)
    plt.style.use('fivethirtyeight')
    plt.plot(range(0,hm_epochs),train_matrix,'b' ,label= 'train')
    plt.plot(range(0, hm_epochs), test_matrix, 'r', label='test')
    plt.legend()
    plt.show()

def seperating_function(dataset):
    array_val_1 = []
    array_val_2 = []
    for data in dataset:
        array_val_1.append(data[1])
        array_val_2.append(data[2])
    return np.array(array_val_1),np.array(array_val_2)

def draw_graph(x,y):
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    dataset = familytree()
    trainset = random_shuffle(89,dataset)
    testset = random_shuffle(15,dataset)
    train_x,train_y = seperating_function(trainset)
    test_x,test_y = seperating_function(testset)
    x = tf.placeholder(tf.float32, [None, train_x.shape[1]], name="Input")
    y = tf.placeholder(tf.float32, [None, train_y.shape[1]], name="Output")
    train(train_x=train_x,train_y=train_y,test_x=test_x,test_y=test_y,n_hidden=6,n_classes=train_y.shape[1],batch_size=32,n_epoches=30)
    print("\n\n Average Train Accuracy : ", np.mean(train_matrix), " Average Test Accuracy : ", np.mean(test_matrix))
    draw_graph(iter_matrix,error_matrix)




