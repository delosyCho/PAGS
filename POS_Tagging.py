import tensorflow as tf
import numpy as np
import POS_Embed
import data_processor

class Word_Embedding_POS_Taggign:
    def max_pool_axb(self, x, a, b):
        return tf.nn.max_pool(x, ksize=[1, a, b, 1],
                              strides=[1, a, b, 1], padding='VALID')

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_k(self, x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    def get_Char_Array(self, words):
        arr = np.zeros(shape=[self.P_Length, self.C_Length, self.Char_Onehot])

        for i in range(self.P_Length):
            ch_arr = [ord(c) for c in words[i]]
            for j in range(len(ch_arr)):
                if j < self.C_Length:
                    arr[i, j, ch_arr[j]] = 1

        return arr

    def set_Char_Batch(self, words_Batch):
        Char_Batch = np.zeros(shape=[self.Batch, self.P_Length, self.C_Length, self.Char_Onehot], dtype='f')

        for i in range(self.Batch):
            Char_Batch[i] = self.get_Char_Array(words_Batch[i])

        return Char_Batch

    def __init__(self, is_Setting_Embedder=True):
        self.vocab_size = 85
        self.cell_Dec_fw = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.cell_Dec_bw = tf.nn.rnn_cell.BasicLSTMCell(128)

        self.cell_Output_fw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)
        self.cell_Output_bw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)

        self.Weight_Hidden = self.weight_variable(shape=[50, 128], name='Weight_Hidden_Tagger')
        self.Bias_Hidden = self.bias_variable(shape=[128], name='Bias_Hiddne_Tagger')

        self.Weight_Embedding = self.weight_variable(shape=[128, 256], name='Weight_Embedding_Tagger')
        self.Bias_Embedding = self.bias_variable(shape=[256], name='Bias_Embedding_Tagger')

        self.Batch = 500
        self.P_Length = 100
        self.C_Length = 20
        self.Char_Onehot = 128
        self.Embedding_Size = 50

        self.X_Input = tf.placeholder(shape=[None, self.P_Length, self.Embedding_Size], dtype=tf.float32, name='Tagger_Input')

        if is_Setting_Embedder:
            self.pos_Embed = POS_Embed.POS_Embedder()
            self.DataHolder = data_processor.Data_holder(is_Just_Embedding=True)
            self.DataHolder.set_batch()
            self.pos_Embed.set_Data_Hlder(self.DataHolder)

    def setEmbedder(self, embedder):
        self.pos_Embed = embedder

    def Model(self, batch=500):
        with tf.variable_scope("Encoding_P") as scope:
            Weight_Hidden = tf.reshape(self.Weight_Hidden, shape=[1, 50, 128])
            Weight_Hidden = tf.tile(Weight_Hidden, [batch, 1, 1])

            Enc_Input = tf.matmul(self.X_Input, Weight_Hidden) + self.Bias_Hidden

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P") as scope:
            Weight_Embedding = tf.reshape(self.Weight_Embedding, shape=[1, 128, 256])
            Weight_Embedding = tf.tile(Weight_Embedding, [batch, 1, 1])
            Dec_Input = tf.matmul(Enc_Input, Weight_Embedding) + self.Bias_Embedding

            scope.reuse_variables()

        with tf.variable_scope("Decoding_P") as scope:
            Dec, encoding = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Dec_fw,
                                                               cell_bw=self.cell_Dec_bw,
                                                               inputs=Dec_Input,
                                                               sequence_length=self.seq_length(Enc_Input),
                                                               dtype=tf.float32)

            Dec_fw, Dec_bw = Dec
            Output_Input = tf.concat([Dec_fw, Dec_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Decoding_2") as scope:
            output, encoding = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Output_fw,
                                                               cell_bw=self.cell_Output_bw,
                                                               inputs=Output_Input,
                                                               sequence_length=self.seq_length(Output_Input),
                                                               dtype=tf.float32)

            output_fw, output_bw = output
            prediction = tf.nn.sigmoid(output_fw + output_bw)

            scope.reuse_variables()

        return prediction

    def Training(self, is_continue=False, training_epoch=500):
        with tf.Session() as sess:
            prediction = self.Model()

            words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()

            tensor_index = tf.placeholder(tf.float32, shape=[self.Batch, self.P_Length, self.vocab_size], name='Attention_Label')

            #Probability = tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=prediction)
            Probability = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_index, logits=prediction)

            loss = tf.reduce_mean(Probability)
            train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            while self.pos_Embed.epoch < training_epoch:
                words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()

                training_feed_dict = {tensor_index: pos_vec, self.X_Input: words}

                sess.run(train_step, feed_dict=training_feed_dict)

                print(self.pos_Embed.epoch, self.pos_Embed.batch_index, sess.run(loss, feed_dict=training_feed_dict))

                if self.pos_Embed.batch_index == 2000:
                    words, labels, char_words, pos_vec = self.pos_Embed.get_Test_Batch()
                    wrong_cases = self.pos_Embed.wrong_case()

                    test_result = sess.run(prediction, feed_dict={self.X_Input: words})

                    result = 0
                    wrong = 0

                    for i in range(self.Batch):

                        for j in range(100):
                            if labels[i, j] != -1:
                                max = -9999
                                index = 0

                                for k in range(self.vocab_size):
                                    if max < test_result[i, j, k]:
                                        max = test_result[i, j, k]
                                        index = k

                                if labels[i, j] == index:
                                    result = result + 1
                                else:
                                    wrong = wrong + 1


                    print(result, " ... ", wrong)

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

    def Test_Prop(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess,
                                      'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            """
            for i in range(15):
                print(labels[i])
                print(self.pos_Embed.POS_data_Tags[i])
            """
            prediction = self.Model()

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()

            test_result = sess.run(prediction, feed_dict={self.X_Input: words})
            print(test_result[0, 0])
            print()
            print(test_result[0, 1])
            print()
            print(test_result[0, 2])
            result = 0
            wrong = 0

            for i in range(self.Batch):
                sentence = ''

                for j in range(100):
                    if labels[i, j] != -1:
                        max_Case = -9999
                        index = 0

                        for k in range(self.vocab_size):
                            if max_Case < test_result[i, j, k]:
                                max_Case = test_result[i, j, k]
                                index = k

                        sentence = sentence + ' [' + self.pos_Embed.pos_dict[int(labels[i, j])] + '] {' + self.pos_Embed.pos_dict[index] + '}  '
                #print(sentence)

    def Prop(self, words, batch):
        with tf.Session() as sess:

            prediction = self.Model(batch)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.import_meta_graph('C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf.meta')
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            test_result = sess.run(prediction, feed_dict={self.X_Input: words})

            return test_result

class POS_Modeling:
    def max_pool_axb(self, x, a, b):
        return tf.nn.max_pool(x, ksize=[1, a, b, 1],
                              strides=[1, a, b, 1], padding='VALID')

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_k(self, x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    def get_Char_Array(self, words):
        arr = np.zeros(shape=[self.P_Length, self.C_Length, self.Char_Onehot])

        for i in range(self.P_Length):
            ch_arr = [ord(c) for c in words[i]]
            for j in range(len(ch_arr)):
                if j < self.C_Length:
                    arr[i, j, ch_arr[j]] = 1

        return arr

    def set_Char_Batch(self, words_Batch):
        Char_Batch = np.zeros(shape=[self.Batch, self.P_Length, self.C_Length, self.Char_Onehot], dtype='f')

        for i in range(self.Batch):
            Char_Batch[i] = self.get_Char_Array(words_Batch[i])

        return Char_Batch

    def __init__(self, is_Setting_Embedder=True):
        self.Batch = 500
        self.P_Length = 100
        self.C_Length = 20
        self.Char_Onehot = 128
        self.Embedding_Size = 50
        self.vocab_size = 85

        self.cell_Output_fw = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_Output_bw = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.Weight_Embedding = self.weight_variable(shape=[self.vocab_size, 128], name='Weight_Embedding_Modler')
        self.Bias_Embedding = self.bias_variable(shape=[128], name='Bias_Embedding_Modler')

        self.Weight_Output = self.weight_variable(shape=[self.P_Length, 1], name='Weight_Output_Modler')
        self.Bias_Output = self.bias_variable(shape=[1], name='Bias_Output_Modler')

        if is_Setting_Embedder:
            self.pos_Embed = POS_Embed.POS_Embedder()

        self.X_Input = tf.placeholder(dtype=tf.float32, shape=[None, self.P_Length, self.vocab_size], name='Modler_Input')

    def setEmbedder(self, embedder):
        self.pos_Embed = embedder

    def Modeling(self, batch=500):
        with tf.variable_scope("Encoding_POS_M") as scope:
            Weight = tf.reshape(self.Weight_Embedding, shape=[1, self.vocab_size, 128], name='Weight_Res1')
            Weight = tf.tile(Weight, [batch, 1, 1], name='Weight_tile_1')
            Enc_Input = tf.matmul(self.X_Input, Weight) + self.Bias_Embedding

            scope.reuse_variables()

        with tf.variable_scope("Encoding_POS_") as scope:
            output, encoding = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Output_fw,
                                                               cell_bw=self.cell_Output_bw,
                                                               inputs=Enc_Input,
                                                               sequence_length=self.seq_length(Enc_Input),
                                                               dtype=tf.float32)

            output_fw, output_bw = output
            output_ = output_fw + output_bw
            output_ = tf.reshape(output_, shape=[batch, self.P_Length], name='Output_Res_2')

            prediction = tf.nn.sigmoid(tf.matmul(output_, self.Weight_Output), name='Output_Sigmoid')
            print("Check", prediction)
            scope.reuse_variables()

        return prediction

    def Test_Prop(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf')

            prediction = self.Modeling()

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf')

            words, labels, char_words, pos_vec = self.pos_Embed.get_Test_Batch()
            test_result = sess.run(prediction, feed_dict={self.X_Input: pos_vec})

            result = 0

            for i in range(len(test_result)):
                if test_result[i] > 0.8:
                    result = result + 1

            print(result, "/", len(test_result))
            for i in range(500):
                print(labels[i])

    def Training(self, is_continue=False, training_epoch=500):
        with tf.Session() as sess:
            prediction = self.Modeling()

            words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()

            label_arr = np.ones(shape=[self.Batch, 1], dtype='f')
            label_ = tf.placeholder(dtype=tf.float32, shape=[self.Batch, 1])

            wrong_label_arr = np.zeros(shape=[self.Batch, 1], dtype='f')

            tensor_index = tf.placeholder(tf.int32, shape=[self.Batch, self.P_Length], name='Attention_Label')

            tensor_Label = tf.one_hot(tensor_index, self.vocab_size, 1, 0)
            tensor_Label_ = tf.cast(tensor_Label, tf.float32)
            tensor_Label_ = tf.reshape(tensor_Label_, [self.Batch, self.P_Length, self.vocab_size])
            sess.run(tf.initialize_all_variables())
            Probability = (label_ - prediction) * (label_ - prediction) / 2
            #Probability = tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=prediction)

            loss = tf.reduce_mean(Probability)

            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf')

            while self.pos_Embed.epoch < training_epoch:
                words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()
                wrong_cases = self.pos_Embed.wrong_case()

                training_feed_dict = {label_: label_arr, self.X_Input: pos_vec}
                training_feed_dict_ = {label_: wrong_label_arr, self.X_Input: wrong_cases}

                sess.run(train_step, feed_dict=training_feed_dict)
                sess.run(train_step, feed_dict=training_feed_dict_)

                print(self.pos_Embed.epoch, self.pos_Embed.batch_index, sess.run(loss, feed_dict=training_feed_dict))

                if self.pos_Embed.batch_index == 1000:
                    words, labels, char_words, pos_vec = self.pos_Embed.get_Test_Batch()
                    wrong_cases = self.pos_Embed.wrong_case()

                    test_result = sess.run(prediction, feed_dict={self.X_Input: pos_vec})

                    result = 0

                    for i in range(len(test_result)):
                        if test_result[i] > 0.8:
                            result = result + 1

                    print(result, "/", len(test_result))

                    test_result = sess.run(prediction, feed_dict={self.X_Input: wrong_cases})

                    result = 0

                    for i in range(len(test_result)):
                        if test_result[i] > 0.8:
                            result = result + 1

                    print("Wrong Case: ", result, "/", len(test_result))

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf')

    def Prop_Modeling(self, pos_vec, batch):
        #tf.reset_default_graph()
        #graph = tf.Graph()

        #sess = tf.InteractiveSession(graph=graph)

        # tf.reset_default_graph()
        # self.X_Input = tf.placeholder(dtype=tf.float32, shape=[count, self.P_Length, self.vocab_size])
        with tf.Session() as sess:
            prediction = self.Modeling(batch)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.import_meta_graph('C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf.meta')
            saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Modeling/POS_Modeling.ckpf')

            prop_feed_dict = {self.X_Input: pos_vec}
            prop_result = sess.run(prediction, feed_dict=prop_feed_dict)

            return prop_result

class Char_Level_POS_Tagging:

    def max_pool_axb(self, x, a, b):
        return tf.nn.max_pool(x, ksize=[1, a, b, 1],
                              strides=[1, a, b, 1], padding='VALID')

    def seq_length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_k(self, x, k):
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    def get_Char_Array(self, words):
        arr = np.zeros(shape=[self.P_Length, self.C_Length, self.Char_Onehot])

        for i in range(self.P_Length):
            ch_arr = [ord(c) for c in words[i]]
            for j in range(len(ch_arr)):
                if j < self.C_Length:
                    arr[i, j, ch_arr[j]] = 1

        return arr

    def set_Char_Batch(self, words_Batch):
        Char_Batch = np.zeros(shape=[self.Batch, self.P_Length, self.C_Length, self.Char_Onehot], dtype='f')

        for i in range(self.Batch):
            Char_Batch[i] = self.get_Char_Array(words_Batch[i])

        return Char_Batch

    def __init__(self):
        self.batch = 500
        self.Batch = self.batch
        self.P_Length = 100
        self.C_Length = 20
        self.Char_Onehot = 128
        self.Embedding_Size = 50
        self.vocab_size = 85

        self.W_Conv2 = self.weight_variable(shape=[2, self.Char_Onehot, 1, 3])
        self.W_Conv3 = self.weight_variable(shape=[3, self.Char_Onehot, 1, 4])
        self.W_Conv4 = self.weight_variable(shape=[4, self.Char_Onehot, 1, 5])
        self.W_Conv5 = self.weight_variable(shape=[5, self.Char_Onehot, 1, 6])

        self.B_Conv2 = self.bias_variable(shape=[3])
        self.B_Conv3 = self.bias_variable(shape=[4])
        self.B_Conv4 = self.bias_variable(shape=[5])
        self.B_Conv5 = self.bias_variable(shape=[6])

        self.Weight_Embedding = self.weight_variable(shape=[18, 32])
        self.Bias_Embedding = self.bias_variable(shape=[32])

        self.Weight_Embedding_2 = self.weight_variable(shape=[32, 64])
        self.Bias_Embedding_2 = self.bias_variable(shape=[64])

        self.cell_Output_fw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)
        self.cell_Output_bw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)

        self.X_Input = tf.placeholder(shape=[self.Batch, self.P_Length, self.C_Length, self.Char_Onehot],
                                      dtype=tf.float32)

        self.pos_Embed = POS_Embed.POS_Embedder()

    def Model(self):
        with tf.variable_scope("Encoding_P") as scope:
            X_Input_ = tf.reshape(self.X_Input, shape=[-1, self.C_Length, self.Char_Onehot, 1])

            X_conv2 = self.conv2d(X_Input_, self.W_Conv2) + self.B_Conv2
            print(X_conv2)
            X_conv2 = tf.reshape(self.max_pool_k(X_conv2, self.C_Length - 1), shape=[self.Batch, self.P_Length, 3])

            X_conv3 = self.conv2d(X_Input_, self.W_Conv3) + self.B_Conv3
            X_conv3 = tf.reshape(self.max_pool_k(X_conv3, self.C_Length - 2), shape=[self.Batch, self.P_Length, 4])

            X_conv4 = self.conv2d(X_Input_, self.W_Conv4) + self.B_Conv4
            X_conv4 = tf.reshape(self.max_pool_k(X_conv4, self.C_Length - 3), shape=[self.Batch, self.P_Length, 5])

            X_conv5 = self.conv2d(X_Input_, self.W_Conv5) + self.B_Conv5
            X_conv5 = tf.reshape(self.max_pool_k(X_conv5, self.C_Length - 4), shape=[self.Batch, self.P_Length, 6])

            X_conv = tf.concat([X_conv2, X_conv3, X_conv4, X_conv5], axis=2)

            scope.reuse_variables()
        with tf.variable_scope("Encoding_P") as scope:
            Weight_Embedding = tf.reshape(self.Weight_Embedding, shape=[1, 18, 32])
            Weight_Embedding = tf.tile(Weight_Embedding, [self.Batch, 1, 1])

            Weight_Embedding_2 = tf.reshape(self.Weight_Embedding_2, shape=[1, 32, 64])
            Weight_Embedding_2 = tf.tile(Weight_Embedding_2, [self.Batch, 1, 1])

            X_S = tf.matmul(X_conv, Weight_Embedding) + self.Bias_Embedding
            X_M = tf.matmul(X_S, Weight_Embedding_2) + self.Bias_Embedding_2

            scope.reuse_variables()
        with tf.variable_scope("Encoding_P") as scope:
            output, encoding = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Output_fw,
                                                                     cell_bw=self.cell_Output_bw,
                                                                     inputs=X_M,
                                                                     sequence_length=self.seq_length(X_M),
                                                                     dtype=tf.float32)

            output_fw, output_bw = output
            prediction = output_fw + output_bw

            scope.reuse_variables()

        return prediction

    def Training(self, is_continue=False, training_epoch=500):
        with tf.Session() as sess:
            prediction = self.Model()

            words, labels, char_words = self.pos_Embed.get_Next_Batch()

            tensor_index = tf.placeholder(tf.int32, shape=[self.batch, self.P_Length], name='Attention_Label')

            tensor_Label = tf.one_hot(tensor_index, self.vocab_size, 1, 0)
            tensor_Label_ = tf.cast(tensor_Label, tf.float32)
            tensor_Label_ = tf.reshape(tensor_Label_, [self.batch, self.P_Length, self.vocab_size])

            Probability = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_Label_, logits=prediction)
            loss = tf.reduce_mean(Probability)

            train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Char_level/POS_Char_Level.ckpf')

            while self.pos_Embed.Batch_Index < training_epoch:
                words, labels, char_words = self.pos_Embed.get_Next_Batch()

                training_feed_dict = {tensor_index: labels, self.X_Input: char_words}

                sess.run(train_step, feed_dict=training_feed_dict)

                if self.pos_Embed.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, sess.run(loss, feed_dict=training_feed_dict))
                    #print(self.attention_Label)

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Char_level/POS_Char_Level.ckpf')
