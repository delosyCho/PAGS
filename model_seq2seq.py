import tensorflow as tf
import numpy
import data_processor as dp

class Seq2Seq_QA:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
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

        self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

        self.cell_Qr_f = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_Pr_f = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f2 = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b2 = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f3 = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b3 = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_output_f = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_b = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.cell_Qr_f_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_Pr_f_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f2_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b2_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_stack_f3_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
        self.cell_stack_b3_ = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

        self.cell_output_f_ = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_b_ = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.output_Start = None
        self.output_Stop = None

        self.x_q_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, self.q_length, self.embedding_size])
        self.x_p_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, self.p_length, self.embedding_size])

    def set_batch(self, q, p, sta, sto):
        self.question = q
        self.paragraph = p
        self.start_index = sta
        self.stop_index = sto

    def predict(self):

        with tf.variable_scope("Model") as scope:
            X_Q = tf.unstack(self.x_q_holer, axis=1)
            X_P = tf.unstack(self.x_p_holer, axis=1)

            output_Qr, encoding_Qr = tf.nn.static_rnn(self.cell_Qr_f, inputs=X_Q, dtype=tf.float32)
            output_Qr = tf.stack(output_Qr, axis=1)

            scope.reuse_variables()
        with tf.variable_scope("Model2") as scope:
            output_Qr_st, encoding_Qr = tf.nn.static_rnn(self.cell_Qr_f_, inputs=X_Q, dtype=tf.float32)
            output_Qr_st = tf.stack(output_Qr_st, axis=1)

            scope.reuse_variables()

        with tf.variable_scope("filter_convolution_layer") as scope:
            output_Pr, encoding_Pr = tf.nn.static_rnn(self.cell_Pr_f, inputs=X_P, dtype=tf.float32)
            output_Pr = tf.stack(output_Pr, axis=1)
            H_P_ = tf.reshape(output_Pr, shape=[self.batch, self.p_length, 1, self.embedding_size])
            H_P = tf.transpose(H_P_, perm=[0, 3, 2, 1])

            H_Q_ = tf.reshape(output_Qr, shape=[self.batch, 1, self.q_length, self.embedding_size])
            H_Q = tf.transpose(H_Q_, perm=[0, 3, 2, 1])

            H_Q__ = tf.reshape(output_Qr, shape=[self.batch, self.q_length, 1, self.embedding_size])
            H_Q__ = tf.transpose(H_Q__, perm=[0, 3, 2, 1])

            H_C_ = tf.matmul(H_Q, H_P)
            H_C = tf.matmul(H_Q__, H_C_)
            H_C = tf.reshape(H_C, shape=[self.batch, self.embedding_size, self.p_length])
            H_C = tf.transpose(H_C, perm=[0, 2, 1])

            H_C_H_P = tf.concat([H_C, output_Pr], axis=2)
            H_C_H_P = tf.unstack(H_C_H_P, axis=1)

            scope.reuse_variables()

        with tf.variable_scope("filter_convolution_layer2") as scope:
            output_Pr_st, encoding_Pr = tf.nn.static_rnn(self.cell_Pr_f_, inputs=X_P, dtype=tf.float32)
            output_Pr_st = tf.stack(output_Pr_st, axis=1)
            H_P__st = tf.reshape(output_Pr_st, shape=[self.batch, self.p_length, 1, self.embedding_size])
            H_P_st = tf.transpose(H_P__st, perm=[0, 3, 2, 1])

            H_Q__st = tf.reshape(output_Qr_st, shape=[self.batch, 1, self.q_length, self.embedding_size])
            H_Q_st = tf.transpose(H_Q__st, perm=[0, 3, 2, 1])

            H_Q___st = tf.reshape(output_Qr_st, shape=[self.batch, self.q_length, 1, self.embedding_size])
            H_Q___st = tf.transpose(H_Q___st, perm=[0, 3, 2, 1])

            H_C__st = tf.matmul(H_Q_st, H_P_st)
            H_C_st = tf.matmul(H_Q___st, H_C__st)
            H_C_st = tf.reshape(H_C_st, shape=[self.batch, self.embedding_size, self.p_length])
            H_C_st = tf.transpose(H_C_st, perm=[0, 2, 1])

            H_C_H_P_st = tf.concat([H_C_st, output_Pr], axis=2)
            H_C_H_P_st = tf.unstack(H_C_H_P_st, axis=1)

            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Start1") as scope:
            H_O, encoding_H_O = tf.nn.static_rnn(self.cell_stack_f, inputs=H_C_H_P, dtype=tf.float32)
            H_O = tf.stack(H_O, axis=1)
            H_O_stack_f = tf.unstack(H_O, axis=1)
            #H_O_Stack_f = tf.reshape(H_O, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Start2") as scope:
            H_O, encoding_H_O = tf.nn.static_rnn(self.cell_stack_f2, inputs=H_O_stack_f, dtype=tf.float32)
            H_O = tf.stack(H_O, axis=1)
            H_O_stack_f2 = tf.unstack(H_O, axis=1)
            #H_O_Stack_f = tf.reshape(H_O, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Start3") as scope:
            H_O, encoding_H_O = tf.nn.static_rnn(self.cell_stack_f3, inputs=H_O_stack_f2, dtype=tf.float32)
            H_O = tf.stack(H_O, axis=1)
            H_O_stack_f3 = tf.unstack(H_O, axis=1)
            #H_O_Stack_f = tf.reshape(H_O, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Start") as scope:
            H_O_Start, encoding_H_O = tf.nn.static_rnn(self.cell_output_f, inputs=H_O_stack_f3, dtype=tf.float32)
            H_O_Start = tf.stack(H_O_Start, axis=1)
            H_O_Start = tf.reshape(H_O_Start, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Stop1") as scope:
            H_O_Stack_b, encoding_H_O_ = tf.nn.static_rnn(self.cell_stack_b, inputs=H_C_H_P_st, dtype=tf.float32)
            H_O_Stack_b = tf.stack(H_O_Stack_b, axis=1)
            H_O_Stack_b = tf.unstack(H_O_Stack_b, axis=1)
            #H_O_Stack_b = tf.reshape(H_O_Stack_b, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Stop2") as scope:
            H_O_Stack_b, encoding_H_O_ = tf.nn.static_rnn(self.cell_stack_b2, inputs=H_O_Stack_b, dtype=tf.float32)
            H_O_Stack_b = tf.stack(H_O_Stack_b, axis=1)
            H_O_Stack_b2 = tf.unstack(H_O_Stack_b, axis=1)
            #H_O_Stack_b = tf.reshape(H_O_Stack_b, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Stop3") as scope:
            H_O_Stack_b, encoding_H_O_ = tf.nn.static_rnn(self.cell_stack_b3, inputs=H_O_Stack_b2, dtype=tf.float32)
            H_O_Stack_b = tf.stack(H_O_Stack_b, axis=1)
            H_O_Stack_b3 = tf.unstack(H_O_Stack_b, axis=1)
            #H_O_Stack_b = tf.reshape(H_O_Stack_b, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        with tf.variable_scope("prediction_layer_Stop") as scope:
            H_O_Stack_b, encoding_H_O_ = tf.nn.static_rnn(self.cell_output_f, inputs=H_O_Stack_b3, dtype=tf.float32)
            # H_O_Stop = tf.stack(H_O, axis=1)
            H_O_Stop = tf.reshape(H_O_Stack_b, shape=[self.batch, self.p_length])
            scope.reuse_variables()

        return H_O_Start, H_O_Stop

    def training(self, training_epoch):
        with tf.Session() as sess:
            H_O_Start, H_O_Stop = self.predict()

            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])

            Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

            Start_Label = tf.cast(Start_Label, tf.float32)
            Stop_Label = tf.cast(Stop_Label, tf.float32)

            H_O_Start = tf.reshape(H_O_Start, shape=[self.batch, self.p_length])
            H_O_Stop = tf.reshape(H_O_Stop, shape=[self.batch, self.p_length])

            loss_start = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label, logits=H_O_Start))
            loss_stop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label, logits=H_O_Stop))

            loss = loss_start + loss_stop

            train_step_start = tf.train.AdamOptimizer(0.0005).minimize(loss_start)
            train_step_stop = tf.train.AdamOptimizer(0.0005).minimize(loss_stop)

            sess = tf.Session()
            sess.run(tf.initialize_all_variables())

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

                    output_sta = sess.run(H_O_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(H_O_Stop,
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
                    print("Test Completed.")

                #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

            self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

            output_sta = sess.run(H_O_Start,
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

                if max_sta_index == self.start_index[i]:
                    sta_rate = sta_rate + 1

            print("TestStart")
            print("Start : ", sta_rate, "/", self.batch)
            print("Test Completed.")

        return 0

    def training_cpu(self, training_epoch):
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                H_O_Start, H_O_Stop = self.predict()

                Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])
                Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])

                Start_Label = tf.one_hot(Start_Inp, self.p_length, 1, 0)
                Stop_Label = tf.one_hot(Stop_Inp, self.p_length, 1, 0)

                Start_Label = tf.cast(Start_Label, tf.float32)
                Stop_Label = tf.cast(Stop_Label, tf.float32)

                H_O_Start = tf.reshape(H_O_Start, shape=[self.batch, self.p_length])
                H_O_Stop = tf.reshape(H_O_Stop, shape=[self.batch, self.p_length])

                loss_start = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label, logits=H_O_Start))
                loss_stop = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label, logits=H_O_Stop))

                loss = loss_start + loss_stop

                train_step_start = tf.train.RMSPropOptimizer(0.0005).minimize(loss)
                train_step_stop = tf.train.RMSPropOptimizer(0.0005).minimize(loss)

                sess = tf.Session()
                sess.run(tf.initialize_all_variables())
                while self.dataset.whole_batch_index < training_epoch:
                    self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()
                    sess.run(train_step_start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                              self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sess.run(train_step_stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    if self.dataset.batch_index == self.batch:
                        print(self.dataset.whole_batch_index, self.dataset.batch_index, self.dataset.numberOf_available_question)
                        print("Loss:", sess.run(loss,
                                              feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                         self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                        output_sta = sess.run(H_O_Start,
                                              feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                         self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                        output_sto = sess.run(H_O_Stop,
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

                    #print(sess.run(Start_Label, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])
                    #print(sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index})[1])

                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_test_batch()

                output_sta = sess.run(H_O_Start,
                                      feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                 self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                output_sto = sess.run(H_O_Stop,
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
                print("Stop : ", sto_rate, "/", self.batch)
                print("Test Completed.")


        return 0

    def validation(self):
        return 0


