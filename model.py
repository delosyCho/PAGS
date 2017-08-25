import tensorflow as tf
import numpy
import data_processor as dp

class CAF:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self):
        self.dataset = dp.Data_holder()
        self.dataset.set_batch()

        self.question = 0
        self.paragraph = 0
        self.start_index = 0
        self.stop_index = 0
        self.batch = 1000
        self.p_length = 125
        self.q_length = 40
        self.seq_length = [40] * 1000
        self.seq_length_ = [125] * 1000

        self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()

        self.cell_Qr = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.cell_Pr = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.cell_output_START = tf.nn.rnn_cell.BasicLSTMCell(1)
        self.cell_output_STOP = tf.nn.rnn_cell.BasicLSTMCell(1)

        self.W_conv1 = self.weight_variable([2, 100, 1, 4])
        self.b_conv1 = self.bias_variable([4])

        self.W_conv2 = self.weight_variable([3, 100, 1, 4])
        self.b_conv2 = self.bias_variable([4])

        self.W_conv3 = self.weight_variable([4, 100, 1, 4])
        self.b_conv3 = self.bias_variable([4])

        self.W_conv4 = self.weight_variable([5, 100, 1, 4])
        self.b_conv4 = self.bias_variable([4])

        self.W_conv5 = self.weight_variable([6, 100, 1, 4])
        self.b_conv5 = self.bias_variable([4])

        self.W_fc_Q = self.weight_variable([20, 90])
        self.b_fc = self.bias_variable([90])

        self.output_Start = None
        self.output_Stop = None

        self.x_q_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, 40, 100])
        self.x_p_holer = tf.placeholder(dtype=tf.float32, shape=[self.batch, 125, 100])

    def set_batch(self, q, p, sta, sto):
        self.question = q
        self.paragraph = p
        self.start_index = sta
        self.stop_index = sto

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv2d_padding(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_axb(self, x, a, b):
        return tf.nn.max_pool(x, ksize=[1, a, b, 1],
                              strides=[1, a, b, 1], padding='VALID')

    def conv_each_filter(self, h_prev, packed_inputs):
        h_prev = tf.reshape(h_prev, shape=[125, 100])
        packed_inputs = tf.reshape(packed_inputs, shape=[1, -1])

        print("Arg Check")
        print(packed_inputs)

        input_conv = tf.slice(packed_inputs, [0, 0], [1, 125 * 100])
        filter_conv = tf.slice(packed_inputs, [0, 9], [1, 90])

        input_conv = tf.reshape(input_conv, shape=(1, 125, 100, 1))
        filter_conv = tf.reshape(filter_conv, shape=(3, 30, 1, 1))

        conv = tf.nn.conv2d(input_conv, filter_conv, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.reshape(conv, shape=[125, 100])
        return conv

    def predict(self):

        with tf.variable_scope("Model"):
            one_arr = numpy.ones((self.batch, 1), dtype='f')
            for i in range(self.batch):
                one_arr[i] = i

            x_q = tf.unstack(self.x_q_holer, axis=1)
            x_p = tf.unstack(self.x_p_holer, axis=1)

            output_Qr, encoding_Qr = tf.nn.dynamic_rnn(self.cell_Qr, x_q, self.seq_length, dtype=tf.float32)
            output_Qr = tf.stack(output_Qr, axis=1)
            output_Qr = tf.reshape(output_Qr, shape=[self.batch, 40, 100, 1])

            h_conv1 = tf.nn.relu(self.conv2d(output_Qr, self.W_conv1) + self.b_conv1)
            feature_map1 = self.max_pool_axb(h_conv1, 39, 1)
            feature_map1 = tf.reshape(feature_map1, shape=[self.batch, 4])

            h_conv2 = tf.nn.relu(self.conv2d(output_Qr, self.W_conv2) + self.b_conv2)
            feature_map2 = self.max_pool_axb(h_conv2, 38, 1)
            feature_map2 = tf.reshape(feature_map2, shape=[self.batch, 4])

            h_conv3 = tf.nn.relu(self.conv2d(output_Qr, self.W_conv3) + self.b_conv3)
            feature_map3 = self.max_pool_axb(h_conv3, 37, 1)
            feature_map3 = tf.reshape(feature_map3, shape=[self.batch, 4])

            h_conv4 = tf.nn.relu(self.conv2d(output_Qr, self.W_conv4) + self.b_conv4)
            feature_map4 = self.max_pool_axb(h_conv4, 36, 1)
            feature_map4 = tf.reshape(feature_map4, shape=[self.batch, 4])

            h_conv5 = tf.nn.relu(self.conv2d(output_Qr, self.W_conv5) + self.b_conv5)
            feature_map5 = self.max_pool_axb(h_conv5, 35, 1)
            feature_map5 = tf.reshape(feature_map5, shape=[self.batch, 4])

            feature_maps = tf.concat([feature_map1, feature_map2, feature_map3, feature_map4, feature_map5], 1)
            feature_maps = tf.reshape(feature_maps, shape=(self.batch, -1))

            kernel_filter = tf.nn.relu(tf.matmul(feature_maps, self.W_fc_Q) + self.b_fc)
            kernel_filter = tf.reshape(kernel_filter, shape=[self.batch, 90])
            kernel_filter = tf.sigmoid(kernel_filter)

        with tf.variable_scope("filter_convolution_layer"):
            output_Pr, encoding_Pr = tf.nn.dynamic_rnn(self.cell_Pr, x_p, self.seq_length_, dtype=tf.float32)
            output_Pr = tf.stack(output_Pr, axis=1)
            output_Pr_ = tf.reshape(output_Pr, shape=[self.batch, 125 * 100])

            input_scan = tf.concat([output_Pr_, kernel_filter], 1)
            input_scan = tf.reshape(input_scan, shape=[1, self.batch, -1])

            packed_inputs = tf.unstack(input_scan, axis=0)

            print(input_scan[0])
            print("Set all ready")

            initialState = tf.zeros([125, 100], name="initial_state")
            print(packed_inputs[0])
            print(x_q[0])

            conv_states = tf.scan(self.conv_each_filter, elems=packed_inputs, initializer=initialState, name='states')
            conv_states = tf.reshape(conv_states, shape=[self.batch, 125, 100])
            conv_states_reverse = tf.reverse(tensor=conv_states, axis=[1])
            print(conv_states)
            print(output_Pr)
            conv_states = tf.concat([conv_states, output_Pr], axis=2)
            conv_states_reverse = tf.concat([conv_states_reverse, output_Pr], axis=2)

            conv_states = tf.unstack(conv_states, axis=1)
            conv_states_reverse = tf.unstack(conv_states_reverse, axis=1)

            print("Set scan conv complete")

        with tf.variable_scope("prediction_layer_Start"):

            output_P_Start, Encoding_output_start = tf.nn.dynamic_rnn(
                self.cell_output_START, inputs=conv_states, sequence_length=self.seq_length_, dtype=tf.float32)
            #self.output_Start = tf.nn.softmax(output_P_Start)
            self.output_Start = tf.stack(output_P_Start, axis=1)
        with tf.variable_scope("prediction_layer_Stop"):
            output_P_Stop, Encoding_output_stop = tf.nn.static_rnn(
                self.cell_output_STOP, inputs=conv_states_reverse, sequence_length=self.seq_length_, dtype=tf.float32)
            #self.output_Stop = tf.nn.softmax(output_P_Stop)
            self.output_Stop = tf.stack(output_P_Stop, axis=1)

            print("Graph all ready")

            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            #print(sess.run(output_P_Start).shape)
            print("Complete")

        return self.output_Start, self.output_Stop

    def training(self, training_epoch):
        with tf.Session() as sess:
            output_Start, output_Stop = self.predict()

            Start_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])
            Stop_Inp = tf.placeholder(tf.int32, shape=[self.batch, 1])

            Start_Label = tf.one_hot(Start_Inp, 125, 1, 0)
            Stop_Label = tf.one_hot(Stop_Inp, 125, 1, 0)

            Start_Label = tf.cast(Start_Label, tf.float32)
            Stop_Label = tf.cast(Stop_Label, tf.float32)

            output_Start = tf.reshape(output_Start, shape=[self.batch, 125])
            output_Stop = tf.reshape(output_Stop, shape=[self.batch, 125])

            loss_start = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Start_Label, logits=output_Start))
            loss_stop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Stop_Label, logits=output_Stop))

            loss = loss_start + loss_stop

            train_step = tf.train.RMSPropOptimizer(0.05).minimize(loss_start)
            train_step2 = tf.train.RMSPropOptimizer(0.05).minimize(loss_stop)

            sess.run(tf.initialize_all_variables())

            loss_f = 0.01

            count = 0
            while self.dataset.whole_batch_index < training_epoch:
                self.paragraph, self.question, self.start_index, self.stop_index = self.dataset.get_next_batch()
                sess.run(train_step, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                          self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                sess.run(train_step2, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                if self.dataset.whole_batch_index % 5 == 0 and self.dataset.batch_index == 1000:
                    print(self.dataset.whole_batch_index, self.dataset.batch_index, self.dataset.numberOf_available_question)
                    print("Loss:", sess.run(loss,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph}))

                    output_sta = sess.run(output_Start,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
                    output_sto = sess.run(output_Stop,
                                          feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                     self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

                    sta_rate = 0
                    sto_rate = 0

                    for i in range(self.batch):
                        max_sta = -999.99
                        max_sta_index = -1
                        max_sto = -999.99
                        max_sto_index = -1

                        for j in range(125):
                            if max_sta < output_sta[i, j]:
                                max_sta = output_sta[i, j]
                                max_sta_index = j

                        for j in range(125):
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
            output_sta = sess.run(output_Start, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                   self.x_q_holer: self.question, self.x_p_holer: self.paragraph})
            output_sto = sess.run(output_Stop, feed_dict={Start_Inp: self.start_index, Stop_Inp: self.stop_index,
                                                       self.x_q_holer: self.question, self.x_p_holer: self.paragraph})

            sta_rate = 0
            sto_rate = 0

            for i in range(self.batch):
                max_sta = -999.99
                max_sta_index = -1
                max_sto = -999.99
                max_sto_index = -1

                for j in range(125):
                    if max_sta < output_sta[i, j]:
                        max_sta = output_sta[i, j]
                        max_sta_index = j

                for j in range(125):
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

        return 0

    def validation(self):
        return 0


