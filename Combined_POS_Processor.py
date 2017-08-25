import tensorflow as tf
import numpy as np
import POS_Embed
import data_processor

class POS_Processor:
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

    def __init__(self, is_Setting_Embedder=True):
        self.Batch = 500
        self.P_Length = 125
        self.C_Length = 20
        self.Char_Onehot = 128
        self.Embedding_Size = 50

        self.vocab_size = 85

        self.cell_Dec_fw = tf.nn.rnn_cell.BasicLSTMCell(128)
        self.cell_Dec_bw = tf.nn.rnn_cell.BasicLSTMCell(128)

        self.cell_Output_fw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)
        self.cell_Output_bw = tf.nn.rnn_cell.BasicLSTMCell(self.vocab_size)

        self.Weight_Hidden = self.weight_variable(shape=[50, 128], name='Weight_Hidden_Tagger')
        self.Bias_Hidden = self.bias_variable(shape=[128], name='Bias_Hiddne_Tagger')

        self.Weight_Embedding = self.weight_variable(shape=[128, 256], name='Weight_Embedding_Tagger')
        self.Bias_Embedding = self.bias_variable(shape=[256], name='Bias_Embedding_Tagger')

        #

        self.cell_Output_fw_ = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_Output_bw_ = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.Weight_Embedding_ = self.weight_variable(shape=[self.vocab_size, 128], name='Weight_Embedding_Modler')
        self.Bias_Embedding_ = self.bias_variable(shape=[128], name='Bias_Embedding_Modler')

        self.Weight_Output_ = self.weight_variable(shape=[self.P_Length, 1], name='Weight_Output_Modler')
        self.Bias_Output_ = self.bias_variable(shape=[1], name='Bias_Output_Modler')

        self.X_input_ = tf.placeholder(dtype=tf.float32, shape=[None, self.P_Length, self.vocab_size], name='Modler_Input')

        #

        self.X_Input = tf.placeholder(shape=[None, self.P_Length, self.Embedding_Size], dtype=tf.float32,
                                      name='Tagger_Input')

        if is_Setting_Embedder:
            self.pos_Embed = POS_Embed.POS_Embedder()
            self.DataHolder = data_processor.Data_holder(is_Just_Embedding=True)
            self.DataHolder.set_batch()
            self.pos_Embed.set_Data_Hlder(self.DataHolder)

    def setEmbedder(self, embedder):
        self.pos_Embed = embedder

    def Modeling(self, batch=500):
        with tf.variable_scope("Encoding_POS_M") as scope:
            Weight = tf.reshape(self.Weight_Embedding_, shape=[1, self.vocab_size, 128], name='Weight_Res1')
            Weight = tf.tile(Weight, [batch, 1, 1], name='Weight_tile_1')
            Enc_Input = tf.matmul(self.X_input_, Weight) + self.Bias_Embedding_

            scope.reuse_variables()

        with tf.variable_scope("Encoding_POS_") as scope:
            output, encoding = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Output_fw_,
                                                               cell_bw=self.cell_Output_bw_,
                                                               inputs=Enc_Input,
                                                               sequence_length=self.seq_length(Enc_Input),
                                                               dtype=tf.float32)

            output_fw, output_bw = output
            output_ = output_fw + output_bw
            output_ = tf.reshape(output_, shape=[batch, self.P_Length], name='Output_Res_2')

            prediction = tf.nn.sigmoid(tf.matmul(output_, self.Weight_Output_), name='Output_Sigmoid')
            scope.reuse_variables()

        return prediction

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
            modeling = self.Modeling()

            tensor_index = tf.placeholder(tf.float32, shape=[self.Batch, self.P_Length, self.vocab_size], name='Attention_Label')

            label_arr = np.ones(shape=[self.Batch, 1], dtype='f')
            label_ = tf.placeholder(dtype=tf.float32, shape=[self.Batch, 1])

            #Probability = tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=prediction)
            Probability_Tagging = tf.nn.softmax_cross_entropy_with_logits(labels=tensor_index, logits=prediction)
            loss_Tagging = tf.reduce_mean(Probability_Tagging)
            train_step = tf.train.AdamOptimizer(0.01).minimize(loss_Tagging)

            Probability_Modeling = (label_ - modeling) * (label_ - modeling) / 2
            loss_Modeling = tf.reduce_mean(Probability_Modeling)
            train_step_ = tf.train.AdamOptimizer(0.01).minimize(loss_Modeling)

            sess.run(tf.initialize_all_variables())

            if is_continue:
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            while self.pos_Embed.epoch < training_epoch:
                words, labels, char_words, pos_vec = self.pos_Embed.get_Next_Batch()
                wrong_cases = self.pos_Embed.wrong_case()

                training_feed_dict = {tensor_index: pos_vec, self.X_Input: words}
                training_feed_dict_ = {label_: label_arr, self.X_input_: pos_vec}
                training_feed_dict_wrong = {label_: label_arr, self.X_input_: wrong_cases}

                sess.run(train_step, feed_dict=training_feed_dict)
                sess.run(train_step_, feed_dict=training_feed_dict_)
                sess.run(train_step_, feed_dict=training_feed_dict_wrong)

                print(self.pos_Embed.epoch, self.pos_Embed.batch_index, sess.run(loss_Modeling, feed_dict=training_feed_dict_))
                print(self.pos_Embed.epoch, self.pos_Embed.batch_index, sess.run(loss_Tagging, feed_dict=training_feed_dict))

                if self.pos_Embed.batch_index == 2000:
                    words, labels, char_words, pos_vec = self.pos_Embed.get_Test_Batch()
                    wrong_cases = self.pos_Embed.wrong_case()

                    test_result = sess.run(prediction, feed_dict={self.X_Input: words})

                    result = 0
                    wrong = 0

                    for i in range(self.Batch):

                        for j in range(self.P_Length):
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

    def Prop_Tagging(self, words, batch):
        with tf.Session() as sess:

            prediction = self.Model(batch)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.import_meta_graph('C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf.meta')
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

            test_result = sess.run(prediction, feed_dict={self.X_Input: words})

            return test_result

    def Prop_Modeling(self, pos_vec, batch):
        #tf.reset_default_graph()
        #graph = tf.Graph()

        #sess = tf.InteractiveSession(graph=graph)

        # tf.reset_default_graph()
        # self.X_Input = tf.placeholder(dtype=tf.float32, shape=[count, self.P_Length, self.vocab_size])
            with tf.Session() as sess:
                prediction = self.Modeling(batch)

                sess.run(tf.global_variables_initializer())

                saver = tf.train.import_meta_graph(
                    'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf.meta')
                saver.restore(sess,
                              'C:/Users/Administrator/Desktop/PAIG_Model_Saver/POS_Tagging_Word_level/POS_Char_Level.ckpf')

                prop_feed_dict = {self.X_input_: pos_vec}
                prop_result = sess.run(prediction, feed_dict=prop_feed_dict)

                return prop_result
