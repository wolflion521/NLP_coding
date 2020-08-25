# 孪生网络的代码大概是什么样
下面这段代码就是nlp里面用双向RNN
然后用tf.get_variable_scope实现了参数的共享
with tf.name_scope('siamese'), tf.variable_scope('rnn'):
            # tf.contrib.rnn.LSTMCell, 返回一个LSTM cell instance
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            # tf.contrib.rnn.MultiRNNCell, 构建多层循环神经网络。
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.keep_prob)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.keep_prob)
            # outputs is a length T list of outputs (one for each input)
            outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.inputs_x1,
                                                                       dtype=tf.float32)
            output_x1 = tf.reduce_mean(outputs_x1, 0)
            # 开启变量重用的开关
            tf.get_variable_scope().reuse_variables()
            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.inputs_x2,
                                                                       dtype=tf.float32)
            # [batch_size, 1, 2 * self.rnn_size]  self.fc_w1: [2 * self.rnn_size, 128]
            output_x2 = tf.reduce_mean(outputs_x2, 0)
            # [batch_size, 1, 128]
            self.logits_1 = tf.matmul(output_x1, self.fc_w1) + self.fc_b1
            self.logits_2 = tf.matmul(output_x2, self.fc_w2) + self.fc_b2
        # [batch_size, 1]
        f_x1x2 = tf.reduce_sum(tf.multiply(self.logits_1, self.logits_2), 1)
        # tf.square()是对参数里的每一个元素求平方
        # [batch_size, 1]
        norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_1), 1))
        norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_2), 1))
        # https://www.cnblogs.com/dsgcBlogs/p/8619566.html
        self.Ew = f_x1x2 / (norm_fx1 * norm_fx2)
        self.Bw = 1 - self.Ew
        self.logits = tf.concat([self.Ew, self.Bw], axis=0)
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits)
        
        
        
        
        
        
        
        
        
        origin_embeddings = tf.placeholder(tf.float32, shape=[None, None, 312], name="origin_embeddings")
        standard_embeddings = tf.placeholder(tf.float32, shape=[None, None, 312], name="standard_embeddings")
        labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        origin_lengths = tf.placeholder(tf.int32, shape=[None], name='origin_lengths')
        standard_lengths = tf.placeholder(tf.int32, shape=[None], name='standard_lengths')
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')

        base_lr = tf.constant(learn_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.train.exponential_decay(base_lr, step_ph, epoches, 0.98, staircase=True)
        # output = word_embeddings
