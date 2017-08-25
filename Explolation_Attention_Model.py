import tensorflow as tf
import numpy
import data_processor as dp
import pickle

class Seq2Seq_QA:

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

    def __init__(self):
        self.dataset = dp.Data_holder()
        self.dataset.set_batch()
        self.dataset.set_sentence_batch()

        self.question = 0
        self.paragraph = 0
        self.start_index = 0
        self.stop_index = 0
        self.batch = 500
        self.p_length = 100
        self.q_length = 30
        self.embedding_size = 50
        self.output_size = 64

        self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

        self.cell_Qr_f = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)
        self.cell_Qr_b = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)

        self.cell_Pr_f = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)
        self.cell_Pr_b = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)

        self.cell_Decoding_f = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)
        self.cell_Decoding_b = tf.nn.rnn_cell.BasicLSTMCell(self.output_size)

        self.cell_output_Start_f = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_Start_b = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_End_f = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_End_b = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.output_Start = None
        self.output_Stop = None

        self.x_q_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, self.q_length, self.embedding_size],
                                        name='x_q_holer')
        self.x_p_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, self.p_length, self.embedding_size],
                                        name='x_p_holer')

        self.attention_Weight = self.weight_variable([256, 128], name='attention_Weight')
        self.attention_bias = self.bias_variable([128], name='attention_bias')

        self.decoding_Weight = self.weight_variable([256, 128], name='decoding_Weight')
        self.decoding_Bias = self.bias_variable([128], name='decoding_Bias')

        self.attention_Weight_Start = self.weight_variable([128, 1], name='attention_Weight_Start')
        self.attention_bias_Start = self.bias_variable([1], name='attention_bias_Start')

        self.attention_Weight_End = self.weight_variable([128, 1], name='attention_Weight_End')
        self.attention_bias_End = self.bias_variable([1], name='attention_bias_End')

    def set_batch(self, q, p, sta, sto):
        self.question = q
        self.paragraph = p
        self.start_index = sta
        self.stop_index = sto

    def predict(self):

        with tf.variable_scope("Encoding_Q") as scope:
            #X_Q = tf.unstack(self.x_q_holer, axis=1)
            #X_P = tf.unstack(self.x_p_holer, axis=1)
            X_Q = self.x_q_holer
            X_P = self.x_p_holer

            output_Qr, encoding_Qr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Qr_f,
                                                                     cell_bw=self.cell_Qr_b,
                                                                     inputs=X_Q,
                                                                     sequence_length=self.seq_length(X_Q),
                                                                     dtype=tf.float32)
            output_Qr_fw, output_Qr_bw = output_Qr
            H_Q = tf.concat([output_Qr_fw, output_Qr_bw], axis=2)

            scope.reuse_variables()

        with tf.variable_scope("Encoding_P") as scope:
            output_Pr, encoding_Pr = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_Pr_f,
                                                                     cell_bw=self.cell_Pr_b,
                                                                     inputs=X_P,
                                                                     sequence_length=self.seq_length(X_P),
                                                                     dtype=tf.float32)

            output_Pr_fw, output_Pr_bw = output_Pr
            H_P = tf.concat([output_Pr_fw, output_Pr_bw], axis=2)

        with tf.variable_scope("Attention_Layer") as scope:
            H_Q_T = tf.transpose(H_Q, perm=[0, 2, 1])

            S = tf.nn.softmax(tf.matmul(H_P, H_Q_T))
            C_P = tf.matmul(S, H_Q)
            A_con = tf.concat([C_P, H_P], axis=2)

            att_Weights = tf.reshape(self.attention_Weight, [1, 256, 128])
            att_Weights = tf.tile(att_Weights, [self.batch, 1, 1])

            A = tf.matmul(A_con, att_Weights) + self.attention_bias

            scope.reuse_variables()

        with tf.variable_scope("decoding_layer") as scope:
            D_in = tf.concat([H_P, A], axis=2)

            dec_Weights = tf.reshape(self.decoding_Weight, shape=[1, 256, 128])
            dec_Weights = tf.tile(dec_Weights, [self.batch, 1, 1])

            K = tf.nn.relu(tf.matmul(D_in, dec_Weights) + self.decoding_Bias)
            decoding_Output, decoding_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell_Decoding_f, cell_bw=self.cell_Decoding_b,
                inputs=K,
                sequence_length=self.seq_length(K),
                dtype=tf.float32)

            decoding_fw, decoding_bw = decoding_Output
            D_out = tf.concat([decoding_fw, decoding_bw], axis=2)

            scope.reuse_variables()
        with tf.variable_scope("output_layer_start") as scope:
            output_Weights_Start = tf.reshape(self.attention_Weight_Start, shape=[1, 128, 1])
            output_Weights_Start = tf.tile(output_Weights_Start, [self.batch, 1, 1])

            output_Weights_Stop = tf.reshape(self.attention_Weight_End, shape=[1, 128, 1])
            output_Weights_Stop = tf.tile(output_Weights_Stop, [self.batch, 1, 1])

            inp_Start = tf.matmul(D_out, output_Weights_Start) + self.attention_bias_Start
            inp_Stop = tf.matmul(D_out, output_Weights_Stop) + self.attention_bias_End

            output_st, enc_st = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_output_Start_f,
                                                                cell_bw=self.cell_output_Start_b,
                                                                inputs=inp_Start,
                                                                sequence_length=self.seq_length(inp_Start),
                                                                dtype=tf.float32)
            scope.reuse_variables()
        with tf.variable_scope("output_layer_stop") as scope:
            output_st_fw, output_st_bw = output_st
            output_Start = tf.add(output_st_fw, output_st_bw)

            output_ed, enc_ed = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_output_End_f,
                                                                cell_bw=self.cell_output_End_b,
                                                                inputs=inp_Start,
                                                                sequence_length=self.seq_length(inp_Stop),
                                                                dtype=tf.float32)
            output_ed_fw, output_ed_bw = output_ed
            output_End = tf.add(output_ed_fw, output_ed_bw)
            scope.reuse_variables()
        return output_Start, output_End

    def training(self, training_epoch):
        with tf.Session() as sess:

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)
            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label_ = tf.cast(Start_Label, tf.float32)
            Stop_Label_ = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
            Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
            print(Probability_Start)
            loss_start = tf.reduce_mean(Probability_Start)
            loss_stop = tf.reduce_mean(Probability_Stop)

            loss = loss_start + loss_stop

            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
            train_step_start = tf.train.AdamOptimizer(0.001).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.001).minimize(loss_stop)

            sess.run(tf.initialize_all_variables())

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

                sess.run(train_step, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, self.dataset.batch_index,
                          self.dataset.numberOf_available_question)

                    print("Loss:", sess.run(loss_start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                    output_sta = sess.run(output_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(output_End,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sta_rate = 0
                    sto_rate = 0

                    for i in range(self.batch):
                        max_sta = -999.99
                        max_sta_index = -1
                        max_sto = -999.99
                        max_sto_index = -1

                        for j in range(self.p_length):
                            if max_sta < output_sta[i, j]:
                                max_sta = output_sta[i, j]
                                max_sta_index = j

                        for j in range(self.p_length):
                            if max_sto < output_sto[i, j]:
                                max_sto = output_sto[i, j]
                                max_sto_index = j

                        if max_sta_index == self.start_index[i]:
                            sta_rate = sta_rate + 1
                        if max_sto_index == self.stop_index[i]:
                            sto_rate = sto_rate + 1

                    print("Start : ", sta_rate, "/", self.batch)
                    print("Stop : ", sto_rate, "/", self.batch)
                    print("Eval Completed.")

                #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

            output_sta = sess.run(output_Start,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            output_sto = sess.run(output_End,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sta_rate = 0
            sto_rate = 0

            for i in range(self.batch):
                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(self.p_length):
                    if max_sta < output_sta[i, j]:
                        max_sta = output_sta[i, j]
                        max_sta_index = j

                for j in range(self.p_length):
                    if max_sto < output_sto[i, j]:
                        max_sto = output_sto[i, j]
                        max_sto_index = j

                if max_sta_index == self.start_index[i]:
                    sta_rate = sta_rate + 1
                if max_sto_index == self.stop_index[i]:
                    sto_rate = sto_rate + 1

            print("TestStart")
            print("Start : ", sta_rate, "/", self.batch)
            print(sess.run(loss,
                           feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                      self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
            print("Test Completed.")

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

        return 0

    def training_continue(self, training_epoch):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)
            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label_ = tf.cast(Start_Label, tf.float32)
            Stop_Label_ = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
            Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
            print(Probability_Start)
            loss_start = tf.reduce_mean(Probability_Start)
            loss_stop = tf.reduce_mean(Probability_Stop)

            loss = loss_start + loss_stop

            #train_step = tf.train.AdamOptimizer(0.000005).minimize(loss)

            train_step_start = tf.train.AdamOptimizer(0.0000012).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.000001).minimize(loss_stop)

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

                sess.run(train_step_start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                #sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                #                                self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, self.dataset.batch_index,
                          self.dataset.numberOf_available_question)

                    print("Loss:", sess.run(loss_start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                    output_sta = sess.run(output_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(output_End,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sta_rate = 0
                    sto_rate = 0

                    for i in range(self.batch):
                        max_sta = -999.99
                        max_sta_index = -1
                        max_sto = -999.99
                        max_sto_index = -1

                        for j in range(self.p_length):
                            if max_sta < output_sta[i, j]:
                                max_sta = output_sta[i, j]
                                max_sta_index = j

                        for j in range(self.p_length):
                            if max_sto < output_sto[i, j]:
                                max_sto = output_sto[i, j]
                                max_sto_index = j

                        if max_sta_index == self.start_index[i]:
                            sta_rate = sta_rate + 1
                        if max_sto_index == self.stop_index[i]:
                            sto_rate = sto_rate + 1

                    print("Start : ", sta_rate, "/", self.batch)
                    print("Stop : ", sto_rate, "/", self.batch)
                    print("Eval Completed.")

                #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

            output_sta = sess.run(output_Start,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            output_sto = sess.run(output_End,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sta_rate = 0
            sto_rate = 0

            for i in range(self.batch):
                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(self.p_length):
                    if max_sta < output_sta[i, j]:
                        max_sta = output_sta[i, j]
                        max_sta_index = j

                for j in range(self.p_length):
                    if max_sto < output_sto[i, j]:
                        max_sto = output_sto[i, j]
                        max_sto_index = j

                if max_sta_index == self.start_index[i]:
                    sta_rate = sta_rate + 1
                if max_sto_index == self.stop_index[i]:
                    sto_rate = sto_rate + 1

            print("TestStart")
            print("Start : ", sta_rate, "/", self.batch)
            print(sess.run(loss,
                           feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                      self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
            print("Test Completed.")

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

        return 0

    def training_sto(self, training_epoch):
        with tf.Session() as sess:

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)
            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label_ = tf.cast(Start_Label, tf.float32)
            Stop_Label_ = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
            Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
            print(Probability_Start)
            loss_start = tf.reduce_mean(Probability_Start)
            loss_stop = tf.reduce_mean(Probability_Stop)

            loss = loss_start + loss_stop

            train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
            train_step_start = tf.train.AdamOptimizer(0.1).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.0005).minimize(loss_stop)

            sess.run(tf.initialize_all_variables())

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

                sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, self.dataset.batch_index,
                          self.dataset.numberOf_available_question)

                    print("Loss:", sess.run(loss_stop,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                    output_sta = sess.run(output_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(output_End,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sta_rate = 0
                    sto_rate = 0

                    for i in range(self.batch):
                        max_sta = -999.99
                        max_sta_index = -1
                        max_sto = -999.99
                        max_sto_index = -1

                        for j in range(self.p_length):
                            if max_sta < output_sta[i, j]:
                                max_sta = output_sta[i, j]
                                max_sta_index = j

                        for j in range(self.p_length):
                            if max_sto < output_sto[i, j]:
                                max_sto = output_sto[i, j]
                                max_sto_index = j

                        if max_sta_index == self.start_index[i]:
                            sta_rate = sta_rate + 1
                        if max_sto_index == self.stop_index[i]:
                            sto_rate = sto_rate + 1

                    print("Start : ", sta_rate, "/", self.batch)
                    print("Stop : ", sto_rate, "/", self.batch)
                    print("Eval Completed.")

                #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

            output_sta = sess.run(output_Start,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            output_sto = sess.run(output_End,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sta_rate = 0
            sto_rate = 0

            for i in range(self.batch):
                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(self.p_length):
                    if max_sta < output_sta[i, j]:
                        max_sta = output_sta[i, j]
                        max_sta_index = j

                for j in range(self.p_length):
                    if max_sto < output_sto[i, j]:
                        max_sto = output_sto[i, j]
                        max_sto_index = j

                if max_sta_index == self.start_index[i]:
                    sta_rate = sta_rate + 1
                if max_sto_index == self.stop_index[i]:
                    sto_rate = sto_rate + 1

            print("TestStart")
            print("Start : ", sta_rate, "/", self.batch)
            print(sess.run(loss,
                           feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                      self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
            print("Test Completed.")

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver_Stop/word2vec.ckpf')

        return 0

    def training_continue_sto(self, training_epoch):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver_Stop/word2vec.ckpf')

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)
            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label_ = tf.cast(Start_Label, tf.float32)
            Stop_Label_ = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
            Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
            print(Probability_Start)
            loss_start = tf.reduce_mean(Probability_Start)
            loss_stop = tf.reduce_mean(Probability_Stop)

            loss = loss_start + loss_stop

            #train_step = tf.train.AdamOptimizer(0.000005).minimize(loss)

            train_step_start = tf.train.AdamOptimizer(0.0000012).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.00001).minimize(loss_stop)

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver_Stop/word2vec.ckpf')

            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

                sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                #sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                #                                self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                if self.dataset.batch_index == self.batch:
                    print(self.dataset.whole_batch_index, self.dataset.batch_index,
                          self.dataset.numberOf_available_question)

                    print("Loss:", sess.run(loss_stop,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                    output_sta = sess.run(output_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(output_End,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sta_rate = 0
                    sto_rate = 0

                    for i in range(self.batch):
                        max_sta = -999.99
                        max_sta_index = -1
                        max_sto = -999.99
                        max_sto_index = -1

                        for j in range(self.p_length):
                            if max_sta < output_sta[i, j]:
                                max_sta = output_sta[i, j]
                                max_sta_index = j

                        for j in range(self.p_length):
                            if max_sto < output_sto[i, j]:
                                max_sto = output_sto[i, j]
                                max_sto_index = j

                        if max_sta_index == self.start_index[i]:
                            sta_rate = sta_rate + 1
                        if max_sto_index == self.stop_index[i]:
                            sto_rate = sto_rate + 1

                    print("Start : ", sta_rate, "/", self.batch)
                    print("Stop : ", sto_rate, "/", self.batch)
                    print("Eval Completed.")

                #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

            output_sta = sess.run(output_Start,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            output_sto = sess.run(output_End,
                                  feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                             self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sta_rate = 0
            sto_rate = 0

            for i in range(self.batch):
                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(self.p_length):
                    if max_sta < output_sta[i, j]:
                        max_sta = output_sta[i, j]
                        max_sta_index = j

                for j in range(self.p_length):
                    if max_sto < output_sto[i, j]:
                        max_sto = output_sto[i, j]
                        max_sto_index = j

                if max_sta_index == self.start_index[i]:
                    sta_rate = sta_rate + 1
                if max_sto_index == self.stop_index[i]:
                    sto_rate = sto_rate + 1

            print("TestStart")
            print("Start : ", sto_rate, "/", self.batch)
            print(sess.run(loss,
                           feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                      self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
            print("Test Completed.")

            saver = tf.train.Saver()
            save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver_Stop/word2vec.ckpf')

        return 0

    def training_continue_cpu(self, training_epoch):
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

                output_Start, output_End = self.predict()
                print(output_Start)
                print(output_End)
                Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
                Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

                Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
                Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

                Start_Label_ = tf.cast(Start_Label, tf.float32)
                Stop_Label_ = tf.cast(Stop_Label, tf.float32)

                output_Start = tf.reshape(output_Start, [self.batch, 100])
                output_End = tf.reshape(output_End, [self.batch, 100])

                Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
                Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

                Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
                Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
                print(Probability_Start)
                loss_start = tf.reduce_mean(Probability_Start)
                loss_stop = tf.reduce_mean(Probability_Stop)

                loss = loss_start + loss_stop

                train_step_start = tf.train.AdamOptimizer(0.00005).minimize(loss_start)
                train_step_stop = tf.train.AdamOptimizer(0.000025).minimize(loss_stop)

                saver = tf.train.Saver()
                save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

                while self.dataset.whole_batch_index < training_epoch:
                    self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

                    sess.run(train_step_start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                              self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    if self.dataset.batch_index == self.batch:
                        print(self.dataset.whole_batch_index, self.dataset.batch_index,
                              self.dataset.numberOf_available_question)

                        print("Loss:", sess.run(loss_start,
                                              feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                         self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                        output_sta = sess.run(output_Start,
                                              feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                         self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                        output_sto = sess.run(output_End,
                                              feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                         self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                        sta_rate = 0
                        sto_rate = 0

                        for i in range(self.batch):
                            max_sta = -999.99
                            max_sta_index = -1
                            max_sto = -999.99
                            max_sto_index = -1

                            for j in range(self.p_length):
                                if max_sta < output_sta[i, j]:
                                    max_sta = output_sta[i, j]
                                    max_sta_index = j

                            for j in range(self.p_length):
                                if max_sto < output_sto[i, j]:
                                    max_sto = output_sto[i, j]
                                    max_sto_index = j

                            if max_sta_index == self.start_index[i]:
                                sta_rate = sta_rate + 1
                            if max_sto_index == self.stop_index[i]:
                                sto_rate = sto_rate + 1

                        print("Start : ", sta_rate, "/", self.batch)
                        print("Stop : ", sto_rate, "/", self.batch)
                        print("Eval Completed.")

                    #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                    #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

                output_sta = sess.run(output_Start,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                output_sto = sess.run(output_End,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                sta_rate = 0
                sto_rate = 0

                for i in range(self.batch):
                    max_sta = -999.99
                    max_sta_index = -1
                    max_sto = -999.99
                    max_sto_index = -1

                    for j in range(self.p_length):
                        if max_sta < output_sta[i, j]:
                            max_sta = output_sta[i, j]
                            max_sta_index = j

                    for j in range(self.p_length):
                        if max_sto < output_sto[i, j]:
                            max_sto = output_sto[i, j]
                            max_sto_index = j

                    if max_sta_index == self.start_index[i]:
                        sta_rate = sta_rate + 1
                    if max_sto_index == self.stop_index[i]:
                        sto_rate = sto_rate + 1

                print("TestStart")
                print("Start : ", sta_rate, "/", self.batch)
                print(sess.run(loss,
                               feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
                print("Test Completed.")

                saver = tf.train.Saver()
                save_path = saver.save(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

        return 0

    def propagate(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)
            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Start_Inp')
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1], name='Stop_Inp')

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label_ = tf.cast(Start_Label, tf.float32)
            Stop_Label_ = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            Start_Label_ = tf.reshape(Start_Label_, [self.batch, 100])
            Stop_Label_ = tf.reshape(Stop_Label_, [self.batch, 100])

            Probability_Start = tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label_, logits=output_Start)
            Probability_Stop = tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label_, logits=output_End)
            print(Probability_Start)
            loss_start = tf.reduce_mean(Probability_Start)
            loss_stop = tf.reduce_mean(Probability_Stop)

            loss = loss_start + loss_stop

            train_step_start = tf.train.AdamOptimizer(0.0005).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.0005).minimize(loss_stop)

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            sess.run(train_step_start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                  self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()
            print("index:", self.dataset.batch_index)
            if True:
                print(self.dataset.whole_batch_index, self.dataset.batch_index,
                      self.dataset.numberOf_available_question)

                print("Loss:", sess.run(loss_start,
                                        feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                   self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                output_sta = sess.run(output_Start,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                output_sto = sess.run(output_End,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                sta_rate = 0
                sto_rate = 0

                for i in range(self.batch):
                    max_sta = -999.99
                    max_sta_index = -1
                    max_sto = -999.99
                    max_sto_index = -1

                    for j in range(self.p_length):
                        if max_sta < output_sta[i, j]:
                            max_sta = output_sta[i, j]
                            max_sta_index = j

                    for j in range(self.p_length):
                        if max_sto < output_sto[i, j]:
                            max_sto = output_sto[i, j]
                            max_sto_index = j

                    print(self.dataset.start_index_batch[i], self.dataset.stop_index_batch[i])
                    print(max_sta_index, max_sto_index)

                    if max_sta_index == self.start_index[i, 0]:
                        sta_rate = sta_rate + 1
                        print("Start:", max_sta_index, max_sta)
                    if max_sto_index == self.stop_index[i, 0]:
                        sto_rate = sto_rate + 1
                        print("Stop:", max_sto_index, max_sto)
                    if max_sta_index == self.start_index[i, 0] and max_sto_index == self.stop_index[i]:
                        print(self.dataset.question_batch[i])
                        print(self.dataset.paragraph_arr.shape, self.dataset.paragraph_index[i])
                        print(self.dataset.paragraph_arr[self.dataset.paragraph_index[i]])
                        print(self.dataset.paragraph_arr[self.dataset.paragraph_index[i]][max_sta_index],
                              self.dataset.paragraph_arr[self.dataset.paragraph_index[i]][max_sto_index])
                        print(self.dataset.paragraph_arr[self.dataset.paragraph_index[i]][self.dataset.start_index_batch[i]],
                              self.dataset.paragraph_arr[self.dataset.paragraph_index[i]][self.dataset.stop_index_batch[i]])

                        print("@@@@@@@@@@@@@@@@@@@@@@", self.dataset.batch_index, i)


                print("Start : ", sta_rate, "/", self.batch)
                print("Stop : ", sto_rate, "/", self.batch)
                print(sess.run(loss,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
                print("Loss:", sess.run(loss_start,
                                        feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                   self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))
                print("Propagate Eval Completed.")


        return 0

    def get_QA_Answer(self, para, qu):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            output_Start, output_End = self.predict()
            print(output_Start)
            print(output_End)

            output_Start = tf.reshape(output_Start, [self.batch, 100])
            output_End = tf.reshape(output_End, [self.batch, 100])

            saver = tf.train.Saver()
            save_path = saver.restore(sess, 'C:/Users/Administrator/Desktop/PAIG_Model_Saver/word2vec.ckpf')

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_propagate_batch(para, qu)
            print("index:", self.dataset.batch_index)
            if self.dataset.batch_index == self.batch:
                print(self.dataset.whole_batch_index, self.dataset.batch_index,
                      self.dataset.numberOf_available_question)

                output_sta = sess.run(output_Start,
                                      feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                output_sto = sess.run(output_End,
                                      feed_dict={self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(self.p_length):
                    if max_sta < output_sta[0, j]:
                        max_sta = output_sta[0, j]
                        max_sta_index = j

                for j in range(self.p_length):
                    if max_sto < output_sto[0, j]:
                        max_sto = output_sto[0, j]
                        max_sto_index = j

                print("Propagate Eval Completed.")

        return max_sta_index, max_sto_index

    def validation(self):
        return 0


