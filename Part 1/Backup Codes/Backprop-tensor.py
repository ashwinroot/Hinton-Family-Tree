import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


error_matrix = []
iter_matrix = []
train_matrix = []
test_matrix = []
tensor_train = []
tensor_test = []
# numpy_test=  [0.78988925348646433, 0.84854388843314188, 0.92314602132895817, 0.94283018867924529, 0.97138638228055787]
# numpy_train = [0.79135453763969055, 0.951737688812477, 0.95157927053911331, 0.96821073314503249, 0.97833353800810508]
def read_data_train(filename):
    dataset = pd.read_csv(filename)
    return np.array(dataset.loc[:,'Temperature':'HumidityRatio']),np.array(dataset.loc[:,'Occupancy':'Occupancy'])

def read_data_test(filename):
    dataset = pd.read_csv(filename)
    return np.array(dataset.loc[:,'Humidity':'Occupancy']),np.array(dataset.loc[:,'Output':'Output'])

def model_build(n_dim,n_hidden,n_class):
    #hidden layer 1 params
    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([n_dim,n_hidden])),
                     'bias' : tf.Variable(tf.random_normal([n_hidden]))}

    #hidden layer 2params
    hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([n_hidden,n_hidden*2])),
                     'bias' : tf.Variable(tf.random_normal([n_hidden*2]))}
    #output layer params
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden, 1])),
                     'bias': tf.Variable(tf.random_normal([1]))}

    #construction of connections
    l1 = tf.add(tf.matmul(x,hidden_layer1['weights']) , hidden_layer1['bias'])
    l1 = tf.nn.sigmoid(l1)
    # l2 = tf.add(tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['bias'])
    # l2 = tf.nn.sigmoid(l2)
    out = tf.matmul(l1,output_layer['weights'])
    return out


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
def train(train_x,train_y,test_x,test_y,n_hidden,n_classes,batch_size):
    prediction = model_build(train_x.shape[1],n_hidden,n_classes)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    cost = tf.reduce_sum(tf.square(prediction - y)/2)
    correct = tf.abs(tf.round(prediction) - y)
    # correct = tf.equal(prediction,y)
    accuracy =  tf.reduce_sum(correct)

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 20
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for epoch in range(0,hm_epochs):
            epoch_loss = 0
            train_acc = 0
            i = 0
            for p in range(int(train_x.shape[0]/batch_size)):
                x_,y_ = next_batch(batch_size,train_x,train_y)
                _,c, acc_i = sess.run((optimizer,cost, accuracy) , feed_dict={x:x_, y: y_})
                train_acc+=acc_i
                epoch_loss+=c
            train_acc = train_x.shape[0] - train_acc
            epoch_loss /= train_x.shape[0]
            test_acc = test_x.shape[0] - sess.run(accuracy,feed_dict={x:test_x,y:test_y})
            print('Epoch: ',epoch,'loss: ',epoch_loss, 'Train-Accuracy: ',train_acc / train_x.shape[0], 'Test-Accuracy: ', test_acc /test_x.shape[0] )
            train_matrix.append(abs(train_acc / train_x.shape[0]))
            test_matrix.append(abs(test_acc /test_x.shape[0]))
            error_matrix.append(epoch_loss)
            iter_matrix.append(epoch)
    fig = plt.figure("Back Prop using Tensorflow")
    plt.style.use('seaborn-paper')
    plt.plot(range(0,hm_epochs),train_matrix,'r' ,label= 'train-tensor')
    plt.plot(range(0, hm_epochs), test_matrix, 'b', label='test-tensor')
    plt.xlabel("#Epochs")
    plt.ylabel("Accuracy")
    plt.title("n_input->6->10->1")
    plt.legend()
    fig.show()
    fig.savefig("6,10,1")

def draw_graph(x,y):
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    n_hidden_layer = [1, 2, 5, 10, 20]
    train_x,train_y = read_data_train('/Users/ashwinsankar/Desktop/DL/HW2/Code and Dataset/train_data.csv')
    test_x,test_y =   read_data_test('/Users/ashwinsankar/Desktop/DL/HW2/Code and Dataset/test_data.csv')
    x = tf.placeholder(tf.float32, [None, train_x.shape[1]], name="Input")
    y = tf.placeholder(tf.float32, [None, train_y.shape[1]], name="Output")
    train_matrix = []
    test_matrix = []
    train(train_x,train_y,test_x,test_y,5,1,100)
    print("Average Train accuracy :" , np.mean(train_matrix) , "Average Test accuracy :", np.mean(test_matrix))
