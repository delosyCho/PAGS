import tensorflow as tf
import numpy
import data_processor as dp
import math
import nltk
import numpy as np
import random
import data_processor
import POS_Tagging

class POS_Embedder:
    def set_Data_Hlder(self, data_holer):
        self.Data_Holder = data_holer

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

    def get_Fianl_POS(self, words, batch, Prop_Result):
        result = np.zeros([13000, self.POS_data_S_Length], dtype='f')
        temp = np.zeros(shape=[self.POS_data_S_Length], dtype='<U20')

        fileName = 'C:\\Users\\Administrator\Desktop\\PAIG_Model_Saver\\POS_Tag_Result_Saver\\pos_tag_result'
        f = open(fileName, 'r')

        lines = f.readlines()

        for p in range(batch):
            temp = words[p]
            self.tagged = self.tagger.tag(temp)

            for i in range(self.POS_data_S_Length):
                if temp[i] != '#@':
                    if self.tagged[i][1] != None:
                        TK = self.tagged[i][1].split('-')

                        if TK[0] != 'None':
                            index = self.pos_dict.index(TK[0]) if TK[0] in self.pos_dict else -1
                            if index != -1:
                                result[p, i] = index
                            else:
                                # Set POS from Training Model
                                result[p, i] = Prop_Result[p, i]
                        else:
                            # Set POS from Training Model
                            result[p, i] = Prop_Result[p, i]
                    else:
                        # Set POS from Training Model
                        result[p, i] = Prop_Result[p, i]

            for i in range(self.POS_data_S_Length):
                is_Wrong = False

                if temp[i] != '#@':
                    if self.tagged[i][1] != None:
                        TK = self.tagged[i][1].split('-')

                        if TK[0] != 'None':
                            index = self.pos_dict.index(TK[0]) if TK[0] in self.pos_dict else -1
                            if index != -1:
                                0
                            else:
                                # Set POS from Training Model
                                is_Wrong = True
                        else:
                            # Set POS from Training Model
                            is_Wrong = True
                    else:
                        # Set POS from Training Model
                        is_Wrong = True

                if is_Wrong:
                    index = self.pos_dict.index(TK[0]) if TK[0] in self.pos_dict else -1
                    result[p, i] = index

        return result

    def read_POS_From_Embed(self):
        result = numpy.zeros(shape=[13000, self.POS_data_S_Length], dtype='i')
        result2 = numpy.zeros(shape=[13000, self.POS_data_S_Length], dtype='i')

        f = open('C:\\Users\\Administrator\Desktop\\PAIG_Model_Saver\\POS_Tag_Result_Saver\\pos_tag_result')
        lines = f.readlines()

        for i in range(len(lines)):
            TK = lines[i].split('@')

            for j in range(self.POS_data_S_Length):
                result[i, j] = int(TK[j].split('#')[0])
                result2[i, j] = int(TK[j].split('#')[1])

        return result, result2

    def save_POS_From_Embed(self):
        batchIndex = 0

        fileName = 'C:\\Users\\Administrator\Desktop\\PAIG_Model_Saver\\POS_Tag_Result_Saver\\pos_tag_result'
        f = open(fileName, 'w')

        words = np.zeros(shape=[1000, self.POS_data_S_Length, 50], dtype='f')
        result_holder = []

        for i in range(self.Data_Holder.numberOf_available_question + 1000):
            words[batchIndex] = self.Data_Holder.get_glove_sequence(self.POS_data_S_Length, self.Data_Holder.paragraph_arr[i])

            batchIndex = batchIndex + 1
            if batchIndex == 1000:
                result_holder = self.POS_Modler.Prop_Tagging(words, 1000)
                print(result_holder.shape)
                line = ''

                for j in range(1000):
                    line = ''

                    for l in range(self.POS_data_S_Length):
                        max_v = -999
                        max_index = -1

                        max_v2 = -999
                        max_index2 = -1

                        for k in range(85):
                            if max_v < result_holder[j, l, k]:
                                max_v = result_holder[j, l, k]
                                max_index = k
                            elif max_v2 < result_holder[j, l, k]:
                                max_v2 = result_holder[j, l, k]
                                max_index2 = k

                        line = line + str(max_index) + '#' + str(max_index2) + '@'

                    f.write(line + '\n')
                print("wrote")
                batchIndex = 0

        f.close()

    def save_Determined_POS(self, words, S_Length, Prop_Result, second_level_POS):
        filneName = 'C:\\Users\\Administrator\Desktop\\PAIG_Model_Saver\\Determined_POS\\D_POS'
        f = open(filneName, 'w')

        ambig_Words = []
        start_index = [0]

        numberOfSenten = 0

        modeling_batch = np.zeros(shape=[4800000, self.POS_data_S_Length], dtype='f')
        modeling_result = np.zeros(shape=[4800000], dtype='f')

        final_Result_arr = np.zeros(shape=[13000, self.POS_data_S_Length], dtype='f')

        batch_s_start_index = []
        batch_s_stop_index = []
        batch_index = []
        batch_s_label = []

        batch_count = 0

        for p in range(13000):

            for i in range(S_Length):
                if words[p, i] == '.':
                    numberOfSenten = numberOfSenten + 1
                    start_index.append((i + 1))
            start_index.append(S_Length)

            result_sentence = np.zeros(shape=[S_Length, self.vocab_size])

            determined_POS = np.zeros(shape=[1, S_Length])

            for k in range(len(start_index) - 1):

                temp = np.zeros(shape=[S_Length], dtype='<U20')
                for i in range(S_Length):
                    temp[i] = '#@'
                for i in range(start_index[k + 1] - start_index[k]):
                    temp[i] = words[p, start_index[k] + i]

                self.tagged = self.tagger.tag(temp)

                non_det_index = []

                for i in range(S_Length):

                    if temp[i] != '#@':
                        if self.tagged[i][1] != None:
                            TK = self.tagged[i][1].split('-')

                            if TK[0] != 'None':
                                index = self.pos_dict.index(TK[0]) if TK[0] in self.pos_dict else -1
                                if index != -1:
                                    determined_POS[0, i] = index
                                else:
                                    #Set POS from Training Model
                                    determined_POS[0, i] = Prop_Result[p, i]
                                    non_det_index.append(i)
                            else:
                                # Set POS from Training Model
                                determined_POS[0, i] = Prop_Result[p, i]
                                non_det_index.append(i)
                        else:
                            # Set POS from Training Model
                            determined_POS[0, i] = Prop_Result[p, i]
                            non_det_index.append(i)

                batch_s_label.append(p)
                batch_index.append(batch_count)
                batch_s_start_index.append(start_index[k])
                batch_s_stop_index.append(start_index[k + 1])

                print(batch_count, len(non_det_index))
                final_Result_arr[p] = determined_POS[0]

                for i in range(len(non_det_index) + 1):
                    modeling_batch[batch_count] = determined_POS[0]
                    if i != len(non_det_index):
                        modeling_batch[batch_count, non_det_index[i]] = second_level_POS[p, non_det_index[i]]
                        batch_count = batch_count + 1

                for j in range(len(non_det_index) + 1):
                    modeling_batch[batch_count] = determined_POS[0]
                    if j != len(non_det_index):
                        modeling_batch[batch_count, non_det_index[j]] = second_level_POS[p, non_det_index[j]]
                        batch_count = batch_count + 1

        batch_s_label.append(p)
        batch_index.append(batch_count)
        batch_s_start_index.append(start_index[k])
        batch_s_stop_index.append(start_index[k + 1])

        print('complete')

        index_count = 0

        batch_arr = np.zeros(shape=[1500, self.POS_data_S_Length, 85], dtype='f')

        for i in range(len(batch_index)):
            if index_count == 1500:
                result_Final_POS = self.POS_Modler.Prop_Modeling(batch_arr, 1500)

                for j in range(1500):
                    max_v = -999
                    max_index = -1

                    for k in range(85):
                        if max_v < result_Final_POS[j, k]:
                            max_index = k

                    for k in range(batch_s_start_index[i - 1500], batch_s_stop_index[i - 1500]):
                        final_Result_arr[batch_s_label[i - 1500], k] = modeling_batch[i - 1500, k]

                batch_count = 0

            for j in range(self.POS_data_S_Length):
                batch_arr[batch_count, j, modeling_batch[i, j]] = 1

            batch_count = batch_count + 1


        f.close()

        return 0

    def get_POS_vector(self, bat_Size, length, words):
        vec = np.zeros(shape=[bat_Size, length, self.vocab_size])

        for i in range(bat_Size):
            for j in range(length):
                0
                #self.pos_dict.index(tagged[i + 1][1]) if tagged[i + 1][1] in self.pos_dict else -1

    def get_POS_Tagging_Test_Data(self):
        batch_sentence, batch_question, batch_start_index, batch_stop_index, attention_L, batch_POS_Embeddings = self.Data_Holder.get_test_batch()

        words = np.zeros(shape=[self.batch_size, self.POS_data_S_Length], dtype='<U20')

        for i in range(self.batch_size):
            words[i] = self.Data_Holder.paragraph_arr[i]

        return batch_sentence, words

    def check_POS_Tagging(self, words):
        tagged = self.tagger.tag(words)
        #print(tagged[0][1])
        """
        if str(tagged[0][1]) == 'None':
            print("Checked!!", tagged[0][0])

            return 1
        """
        count = 0

        for i in range(100):
            if str(words[i]) != '@':
                if str(tagged[i][1]) == 'None':
                    a = self.Data_Holder.get_glove_Test(words[i])
                    if a == 1:
                        count = count + 1

                        print(words[i])
                        #print(words)

        return count

    def wrong_case(self):
        pos_vec = np.zeros(shape=[self.batch_size, self.POS_data_S_Length, self.vocab_size])

        for i in range(self.batch_size):
            for j in range(self.POS_data_S_Length):
                index = random.randrange(0, self.vocab_size)
                pos_vec[i, j, index] = 1

        return pos_vec

    def get_Next_Batch(self):
        words_ebedding = np.zeros(shape=[self.batch_size, self.POS_data_S_Length, 50])
        words = 0
        labels = np.zeros(shape=[self.batch_size, self.POS_data_S_Length])
        pos_vec = np.zeros(shape=[self.batch_size, self.POS_data_S_Length, self.vocab_size])

        if self.batch_index + self.batch_size > self.Whole_Batch_Size:
            self.batch_index = self.batch_size
            self.epoch = self.epoch + 1

        for i in range(self.batch_size):
            a = self.batch_index
            words_ebedding[i] = self.Data_Holder.get_glove_sequence(self.POS_data_S_Length, self.POS_data_Words_Under[a])

            for j in range(self.POS_data_S_Length):
                #print("Check", a, j, i, self.POS_data_Words[a][j])
                arr = [ord(c) for c in self.POS_data_Words[a][j]]
                labels[i, j] = self.POS_data_POS[a][j]
                if self.POS_data_POS[i][j] != -1:
                    pos_vec[i, j, self.POS_data_POS[a][j]] = 1

                for k in range(len(arr)):
                    if k < 20:
                        #words[i, j, k, arr[k]] = 1
                        0

            self.batch_index = self.batch_index + 1

        return words_ebedding, labels, words, pos_vec

    def get_Test_Batch(self):
        words_ebedding = np.zeros(shape=[self.batch_size, self.POS_data_S_Length, 50])
        words = 0
        labels = np.zeros(shape=[self.batch_size, self.POS_data_S_Length])
        pos_vec = np.zeros(shape=[self.batch_size, self.POS_data_S_Length, self.vocab_size])

        if self.batch_index + self.batch_size > self.Whole_Batch_Size:
            self.batch_index = 0
            self.epoch = self.epoch + 1

        for i in range(self.batch_size):
            words_ebedding[i] = self.Data_Holder.get_glove_sequence(self.POS_data_S_Length, self.POS_data_Words[i])

            for j in range(self.POS_data_S_Length):
                # print("Check", a, j, i, self.POS_data_Words[a][j])
                arr = [ord(c) for c in self.POS_data_Words[i][j]]
                labels[i, j] = self.POS_data_POS[i][j]
                if self.POS_data_POS[i][j] != -1:
                    pos_vec[i, j, self.POS_data_POS[i][j]] = 1

                for k in range(len(arr)):
                    if k < 20:
                        # words[i, j, k, arr[k]] = 1
                        0


        return words_ebedding, labels, words, pos_vec

    def processing_Brown_Corpos(self):
        count = 0
        fileName = ['ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cj', 'ck', 'cl', 'cm', 'cn', 'cp']
        numberOfFiles = [44, 27, 17, 17, 36, 48, 75, 30, 80, 29, 24, 6, 29, 26]

        for k in range(len(numberOfFiles)):
            for i in range(1, numberOfFiles[k]):
                if i < 10:
                    ran = '0' + str(i)
                else:
                    ran = str(i)
                print(fileName[k], numberOfFiles[k], i)
                f = open('C:\\Users\\Administrator\\Desktop\\qadataset\\brown\\' + fileName[k] + str(ran), 'r')
                lines = f.readlines()

                for j in range(len(lines)):
                    if len(lines[j].split(' ')) > 2:
                        TK = lines[j].replace('	', '').replace('``/`` ', '').replace('-nc', '').replace('-tl',
                                                                                                           '').split(
                            ' ')

                        # print(lines[j].replace('	', '').replace('``/`` ', '').replace('-nc', '').replace('-tl', ''))
                        if len(TK) < self.POS_data_S_Length:
                            # print(lines[j].replace('	', '').replace('``/`` ', '').replace('-nc', '').replace('-tl', ''))
                            count = count + 1

        self.Whole_Batch_Size = count
        print("Count", count)
        index = 0

        self.POS_data_POS = np.zeros(shape=[count, self.POS_data_S_Length], dtype='i')
        for i in range(count):
            for j in range(self.POS_data_S_Length):
                self.POS_data_POS[i, j] = -1

        self.POS_data_Words = np.zeros(shape=[count, self.POS_data_S_Length], dtype='<U20')
        self.POS_data_Words_Under = np.zeros(shape=[count, self.POS_data_S_Length], dtype='<U20')
        self.POS_data_Tags = np.zeros(shape=[count, self.POS_data_S_Length], dtype='<U20')

        for i in range(count):
            for j in range(self.POS_data_S_Length):
                self.POS_data_Words[i, j] = '@#'
                self.POS_data_Words_Under[i, j] = '@#'

        for k in range(len(numberOfFiles)):
            for i in range(1, numberOfFiles[k]):
                if i < 10:
                    ran = '0' + str(i)
                else:
                    ran = str(i)
                #print(fileName[k], numberOfFiles[k], i)
                f = open('C:\\Users\\Administrator\\Desktop\\qadataset\\brown\\' + fileName[k] + str(ran), 'r')
                lines = f.readlines()

                for j in range(len(lines)):
                    if len(lines[j].split(' ')) > 2:
                        TK = lines[j].replace('	', '').replace('``/`` ', '').replace('-nc', '').replace('-tl', '').split(' ')

                        # print(lines[j].replace('	', '').replace('``/`` ', '').replace('-nc', '').replace('-tl', ''))
                        if len(TK) < self.POS_data_S_Length:
                            for l in range(len(TK)):
                                if len(TK[l].split('/')) > 1:
                                    self.POS_data_POS[index, l] = self.pos_dict.index(TK[l].split('/')[1].upper()) if TK[l].split('/')[1].upper() in self.pos_dict else -1
                                    self.POS_data_Tags[index, l] = TK[l].split('/')[1].upper() + " " + TK[l].split('/')[0]
                                    self.POS_data_Words[index, l] = TK[l].split('/')[0]
                                    self.POS_data_Words_Under[index, l] = TK[l].split('/')[0].lower()
                            index = index + 1


    def input_processing(self):
        pos_dict = []

        pos_filepath = "C:\\Users\\Administrator\\Desktop\\qadataset\\pos.txt"
        pos_f = open(pos_filepath, 'r')

        lines = pos_f.readlines()
        print(len(lines))

        for i in range(len(lines)):
            temp = lines[i].split(' ')[0]
            print(temp)
            pos_dict.append(temp)

        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
        tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

        file_path = "C:\\Users\\Administrator\\Desktop\\qadataset\\text8"

        f = open(file_path, 'r')

        whole_str = str(f.read())
        tokens = whole_str.split(' ')

        tagged = tagger.tag(tokens)

        for i in range(300000):
            input_index = pos_dict.index(tagged[i][1]) if tagged[i][1] in pos_dict else -1

            if input_index != -1:
                if i != 0:
                    thing_index = pos_dict.index(tagged[i - 1][1]) if tagged[i - 1][1] in pos_dict else -1
                    if thing_index != -1:
                        self.training_input.append(input_index)
                        self.training_label.append(thing_index)
                        self.number_of_training_data = self.number_of_training_data + 1
                if i < len(tokens) - 1:
                    thing_index_ = pos_dict.index(tagged[i + 1][1]) if tagged[i + 1][1] in pos_dict else -1
                    if thing_index_ != -1:
                        self.training_input.append(input_index)
                        self.training_label.append(thing_index_)
                        self.number_of_training_data = self.number_of_training_data + 1

        self.pos_dict = pos_dict

        return 0

    def read_POS_file(self):
        self.tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())
        self.embed_Vectors = np.zeros((self.vocab_size, self.embedding_size), dtype='f')

        fileName = 'C:\\Users\\Administrator\\Desktop\\qadataset\\pos_embed'
        f = open(fileName, 'r')

        dataStr = f.readline()
        vec_data = dataStr.split('#')

        for i in range(self.vocab_size):
            #print("Str Vec:", i, vec_data[i])
            TK = vec_data[i].split(' ')

            for j in range(self.embedding_size):
                self.embed_Vectors[i, j] = float(TK[j])

    def get_POS_Embedding(self, index):
        POS_Embed = numpy.zeros((1, self.embedding_size), dtype='f')

        for i in range(self.embedding_size):
            POS_Embed[0, i] = self.embed_Vectors[index, i]

        return POS_Embed

    def pos_tagger(self, words, s_length):
        tagged = self.tagger.tag(words)

        result = numpy.zeros((s_length, self.embedding_size), dtype='f')

        for i in range(s_length):
            input_index = self.pos_dict.index(tagged[i][1]) if tagged[i][1] in self.pos_dict else 0
            result[i] = self.get_POS_Embedding(input_index)

        return result

    def get_POS_batch(self, words, batch_size, s_length):
        result = np.zeros(shape=[batch_size, s_length, self.embedding_size], dtype='f')

        for i in range(batch_size):
            result[i] = self.pos_tagger(words[i], s_length)

    def __init__(self):
        self.tagged = []
        self.POS_data_Words = []
        self.POS_data_Words_Under = []
        self.POS_data_POS = []
        self.POS_data_Tags = []

        self.POS_data_S_Length = 125
        self.Whole_Batch_Size = 0
        self.Batch_Index = 0

        self.pos_dict = []

        self.vocab_size = 85
        self.number_of_training_data = 0
        self.batch_size = 500
        self.batch_index = 500
        self.epoch = 0
        self.word_embedding_size = 50

        self.embedding_size = 128
        self.num_sampled = 32

        self.training_input = []
        self.training_label = []

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                      stddev=1.0 / math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

        con = np.zeros(shape=[self.vocab_size, 1], dtype='i')
        for i in range(self.vocab_size):
            con[i] = i

        self.valid_dataset = tf.constant(con, dtype=tf.int32)
        self.embed_Vectors = []
        self.input_processing()
        self.tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

        self.processing_Brown_Corpos()
        self.Data_Holder = 0

        self.POS_Tagger_By_Embeddings = 0
        self.POS_Modler = 0
        #self.POS_Tagger_By_Embeddings = POS_Tagging.Word_Embedding_POS_Taggign()
        #self.POS_Modler = POS_Tagging.POS_Modeling()

    def set_POS_Tagger(self, Modler):

        self.POS_Modler = Modler

    def get_next_batch(self):
        batch_input = np.zeros(shape=[self.batch_size], dtype=np.int32)
        batch_label = np.zeros(shape=[self.batch_size, 1], dtype=np.int32)

        if self.batch_index + self.batch_size > self.number_of_training_data:
            self.batch_index = 0
            self.epoch = self.epoch + 1

        for i in range(self.batch_size):
            batch_input[i] = self.training_input[self.batch_index + i]
            batch_label[i, 0] = self.training_label[self.batch_index + i]

        self.batch_index = self.batch_index + self.batch_size

        return batch_input, batch_label

    def model(self, epoch=100):
        with tf.Session() as sess:
            with tf.variable_scope("Encoding_Q") as scope:
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=self.nce_weights,
                                   biases=self.nce_biases,
                                   labels=train_labels,
                                   inputs=embed,
                                   num_sampled=self.num_sampled,
                                   num_classes=self.vocab_size))

                optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
                sess.run(tf.initialize_all_variables())
                while self.epoch != epoch:
                    batch_input, batch_label = self.get_next_batch()
                    #print(batch_label.size, batch_input.size)
                    sess.run(optimizer, feed_dict={train_inputs: batch_input, train_labels: batch_label})

                    if self.epoch % 50 == 0 and self.batch_index == self.batch_size:
                        ls = sess.run(loss, feed_dict={train_inputs: batch_input, train_labels: batch_label})
                        print(self.epoch, ls)
                saver = tf.train.Saver()
                save_path = \
                    saver.save(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')

        return 0


    def model_continue(self, epoch=100):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = \
                saver.restore(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')

            with tf.variable_scope("Encoding_Q") as scope:
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=self.nce_weights,
                                   biases=self.nce_biases,
                                   labels=train_labels,
                                   inputs=embed,
                                   num_sampled=self.num_sampled,
                                   num_classes=self.vocab_size))

                optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
                sess.run(tf.initialize_all_variables())

                saver = tf.train.Saver()
                save_path = \
                    saver.restore(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')

                while self.epoch != epoch:
                    batch_input, batch_label = self.get_next_batch()
                    #print(batch_label.size, batch_input.size)
                    sess.run(optimizer, feed_dict={train_inputs: batch_input, train_labels: batch_label})

                    if self.epoch % 50 == 0 and self.batch_index == self.batch_size:
                        ls = sess.run(loss, feed_dict={train_inputs: batch_input, train_labels: batch_label})
                        print(self.epoch, ls)
                saver = tf.train.Saver()
                save_path = \
                    saver.save(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')

        return 0

    def get_embedding(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = \
                saver.restore(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')
            embed = tf.nn.embedding_lookup(self.embeddings, self.valid_dataset)

            embed_result = sess.run(embed)

            return embed_result

    def validate(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = \
                saver.restore(sess, 'C:\\Users\\Administrator\\Desktop\\PAIG_Model_Saver\\POS_Vector\\pos_vec.ckpf')

            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
            valid_embeddings = tf.reshape(valid_embeddings, shape=[self.vocab_size, 128])
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            sim = similarity.eval()

            for i in range(self.vocab_size):
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                print(self.pos_dict[i], "  :     ", self.pos_dict[nearest[0]], self.pos_dict[nearest[1]], self.pos_dict[nearest[2]],
                      self.pos_dict[nearest[3]], self.pos_dict[nearest[4]])


            return  valid_embeddings