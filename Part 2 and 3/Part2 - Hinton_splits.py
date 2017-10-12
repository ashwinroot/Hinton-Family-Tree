import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

error_matrix = []
iter_matrix = []
train_matrix = []
test_matrix = []

names = [ "Christopher", "Andrew", "Arthur", "James", "Charles", "Colin", "Penelope", "Christine", "Victoria", "Jennifer", "Margaret", "Charlotte", "Roberto", "Pierro", "Emilio", "Marco", "Tomaso", "Alfonso", "Maria", "Francesca", "Lucia", "Angela", "Gina", "Sophia"]

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
                  bitvec(p1,len(names)),
                  bitvec(r,len(relations)),
                  bitvec(p2,len(names)) ]
                  #bitvec(p2,len(names)) ]
            dataset.append(d)

    return dataset

def random_shuffle(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    return data_shuffle

def weights_graph(unit,layer_number):
    fig =plt.figure()
    plt.style.use('seaborn-paper')
    plt.barh(range(0, len(names)), unit)
    plt.yticks(range(0, len(names)), names)
    plt.xlabel('Weights')
    word = 'Weight visualisation in unit' , layer_number
    plt.title(word)
    return fig

def seperating_function(dataset):
    array_val_1 = []
    array_val_1_5 = []
    array_val_2 = []
    for data in dataset:
        array_val_1.append(data[1])
        array_val_1_5.append(data[2])
        array_val_2.append(data[3])
    return np.array(array_val_1),np.array(array_val_1_5),np.array(array_val_2)

def model_build(n_person,n_relationship,n_hidden):
    #hidden layer params
    person_1 = {'weights' : tf.Variable(tf.random_normal([n_person,n_hidden])),
                     'bias' : tf.Variable(tf.random_normal([n_hidden]))}
    relationship = {'weights': tf.Variable(tf.random_normal([n_relationship, n_hidden])),
                'bias': tf.Variable(tf.random_normal([n_hidden]))}
    hidden_layer= {'weights': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
                'bias': tf.Variable(tf.random_normal([n_hidden]))}
    person_2 = {'weights': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
                'bias': tf.Variable(tf.random_normal([n_hidden]))}

    #output layer params
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden, n_person])),
                     'bias': tf.Variable(tf.random_normal([n_person]))}
    # print(output_layer)
    #construction of connections
    l1 = tf.add(tf.matmul(x_person,person_1['weights']) , person_1['bias'])
    l1 = tf.nn.sigmoid(l1)
    l2 = tf.add(tf.matmul(x_relationship,relationship['weights']) ,relationship['bias'])
    l2 = tf.nn.sigmoid(l2)

    lx = tf.add(tf.matmul(l1,hidden_layer['weights']),hidden_layer['bias'])
    ly = tf.add(tf.matmul(l2,hidden_layer['weights']),hidden_layer['bias'])
    hl = tf.nn.sigmoid(lx+ly)
    person2 = tf.add(tf.matmul(hl,person_2['weights']),person_2['bias'])
    person2 = tf.nn.sigmoid(person2)
    out = tf.matmul(person2,output_layer['weights'])
    return out,person_1

def next_batch(num, data1,data2, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data1))
    np.random.shuffle(idx)
    idx = idx[:num]
    data1_shuffle = [data1[ i] for i in idx]
    data2_shuffle = [data2[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.array(data1_shuffle), np.array(data2_shuffle), np.array(labels_shuffle)


dataset = familytree()

split_train = []
split_test  = []
for _ in range(0,20):
    print("Starting Split ",_)
    error_matrix = []
    iter_matrix = []
    train_matrix = []
    test_matrix = []
    trainset = random_shuffle(89,dataset)
    testset = random_shuffle(15,dataset)
    train_person, train_relationship, train_y = seperating_function(trainset)
    test_person, test_relationship, test_y = seperating_function(testset)

    n_person = train_person.shape[1]
    n_relationship = train_relationship.shape[1]
    n_train = train_person.shape[0]
    n_test = test_person.shape[0]

    x_person = tf.placeholder(tf.float32, [None, n_person] ,name="Local_Person")
    x_relationship = tf.placeholder(tf.float32, [None, n_relationship],name="Local_Relation")
    y_person = tf.placeholder(tf.float32,[None, n_person] ,name="Out_Person")

    hm_epochs = 1000
    batch_size = 32

    prediction,person_hidden = model_build(n_person,n_relationship,6)

    cost = tf.reduce_sum(tf.losses.mean_squared_error(labels=y_person,predictions=prediction))
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

    correct= tf.equal(tf.argmax(prediction,1), tf.argmax(y_person,1))
    accuracy = tf.reduce_sum(tf.cast(correct,tf.float32))



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, hm_epochs):
            epoch_loss = 0
            train_acc = 0
            i = 0
            for p in range(int(n_train/ batch_size)):
                person_,relation_, y_ = next_batch(batch_size, train_person,train_relationship , train_y)
                _, c, acc_i,pred= sess.run((optimizer, cost,accuracy,prediction), feed_dict={x_person: person_ , x_relationship: relation_, y_person: y_})
                train_acc += acc_i
                epoch_loss += c
            test_acc = sess.run(accuracy,feed_dict={x_person: test_person, x_relationship: test_relationship, y_person: test_y})
            train_acc = sess.run(accuracy,feed_dict={x_person: train_person, x_relationship: train_relationship, y_person: train_y})
            # train_acc = accuracy.eval(feed_dict={x_person: train_person, x_relationship: train_relationship, y_person: train_y})
            # test_acc = accuracy.eval(feed_dict={x_person: test_person, x_relationship: test_relationship, y_person: test_y})
            #print("Epoch:",epoch,"epoch_loss:",epoch_loss,"Train Accuracy : ", train_acc/n_train ,"Test accuracy: ", test_acc/n_test)
            train_matrix.append(train_acc*100/n_train)
            test_matrix.append( test_acc*100/n_test)
        fig1 = plt.figure("Family Tree Accuracy")
        plt.title("Hinton Family Tree: Split Cases")
        plt.style.use('seaborn-paper')
        plt.plot(range(0, hm_epochs), train_matrix, 'r', label='train')
        plt.plot(range(0, hm_epochs), test_matrix, 'b', label='test')
        plt.xlabel("#Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        # plt.show()
        fig1.savefig('SplitHinton.png')
        person_hidden = sess.run(person_hidden)
        hidden = np.array(person_hidden['weights'])
        np.savetxt('weights.txt',hidden)
        i=1
        pp = PdfPages('Hinton_weight_3.pdf')
        for unit in hidden.transpose():
            pp.savefig(weights_graph(unit,i))
            i=i+1
        pp.close()
        split_test.append(np.mean(train_matrix))
        split_train.append(np.mean(test_matrix))
        print("Mean train accuracy for first 1000 epochs,Standard Deviation: ",np.mean(train_matrix[:1000]) ,np.std(train_matrix[:1000]),
              "Mean test accuracy for first 1000 epochs,Standard Deviation: ",np.mean(test_matrix[:1000]),np.std(test_matrix[:1000]))
        print("Mean train accuracy for first 500 epochs,Standard Deviation: ", np.mean(train_matrix[:500]),
              np.std(train_matrix[:500]),
              "Mean train Accuracy for last 500 epochs,Standard Deviation: ", np.mean(train_matrix[-500:]),
              np.std(train_matrix[-500:]),
              "Mean test accuracy for first 500 epochs,Standard Deviation: ", np.mean(test_matrix[:500]),
              np.std(test_matrix[:500]),
              "Mean test Accuracy for last 500 epochs,Standard Deviation: ", np.mean(test_matrix[-500:]),
              np.std(test_matrix[-500:]))
print("\n \n Over 20 splits: Mean Train accuracy:  ",np.mean(split_train),"Mean Test Accuracy: ",np.mean(split_test),"Standard Deviation:",np.std(split_test))