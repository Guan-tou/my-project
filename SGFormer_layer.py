import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from spektral.layers import GCNConv
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

tf.random.set_seed(42)
init = tf.keras.initializers.GlorotUniform(seed=42)


class GraphConvLayer(layers.Layer):
    def __init__(self, in_channels, out_channels, use_weight=True, **kwargs):
        super(GraphConvLayer, self).__init__(** kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_weight = use_weight

        if self.use_weight:
            self.W = self.add_weight(
                shape=(in_channels, out_channels),
                initializer=init,
                name='kernel'
            )
        self.bias = self.add_weight(
            shape=(out_channels,),
            initializer='zeros',
            name='bias'
        )

    def build(self, input_shape):
        super(GraphConvLayer, self).build(input_shape)

    def call(self, x, adj_matrix):
        if self.use_weight:
            x = tf.matmul(x, self.W)

        if isinstance(adj_matrix, tf.sparse.SparseTensor):
            x = tf.sparse.sparse_dense_matmul(adj_matrix, x)
        else:
            x = tf.matmul(adj_matrix, x)

        x = tf.nn.bias_add(x, self.bias)

        return x


class GCN(tf.keras.Model):
    def __init__(self, hidden_channels, en_layers, dropout, use_bn, use_act, use_residual=True):
        super(GCN, self).__init__()

        self.convs = []
        self.bns = []

        current_channels = hidden_channels
        for _ in range(en_layers):
            self.convs.append(GCNConv(current_channels, kernel_initializer=init))
        # for _ in range(en_layers):
        #     self.convs.append(GraphConvLayer(current_channels, current_channels))
        # self.bns = [layers.BatchNormalization(axis=-1) for _ in range(en_layers+1)]
        self.bns = [layers.LayerNormalization(axis=-2) for _ in range(en_layers+1)]
        self.fcs = [layers.Dense(hidden_channels, kernel_initializer=init)]
        self.dropout_layer = layers.Dropout(dropout)

        # self.dropout = dropout
        self.use_bn = use_bn
        self.use_act = use_act
        self.residual = use_residual
        self.activation = tf.keras.activations.relu

    def call(self, data):

        x = data
        adj_matrix = data
        layer_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        # x = self.dropout_layer(x)

        layer_.append(x)

        for i in range(len(self.convs)):
            x = self.convs[i]([x, adj_matrix])
            # x = self.convs[i](x, adj_matrix)

            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x)
            # x = self.dropout_layer(x)
            # if self.residual:
            #     x = 0.5*x + 0.5*layer_[-1]
        return x


def full_attention_conv_tf(qs, ks, vs):
    qs = qs / tf.norm(qs, ord=2)  # [N, H, M]
    ks = ks / tf.norm(ks, ord=2)  # [L, H, M]
    N = tf.shape(qs)[0]

    kvs = tf.einsum('lhm,lhd->hmd', ks, vs)
    attention_num = tf.einsum('nhm,hmd->nhd', qs, kvs)  # [N, H, D]
    N = tf.cast(N, tf.float32)
    attention_num += N * vs

    all_ones = tf.ones([tf.shape(ks)[0]])
    ks_sum = tf.einsum('lhm,l->hm', ks, all_ones)
    attention_normalizer = tf.einsum('nhm,hm->nh', qs, ks_sum)  # [N, H]

    attention_normalizer = tf.expand_dims(attention_normalizer, axis=-1)  # [N, H, 1]
    attention_normalizer += tf.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    return attn_output


class TransConvLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, num_heads, use_weight=True):
        super(TransConvLayer, self).__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        self.Wk = layers.Dense(out_channels * num_heads, kernel_initializer=init)
        self.Wq = layers.Dense(out_channels * num_heads, kernel_initializer=init)
        if use_weight:
            self.Wv = layers.Dense(out_channels * num_heads, kernel_initializer=init)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def call(self, query_input, source_input):
        query = self.Wq(query_input)
        key = self.Wk(source_input)
        if self.use_weight:
            value = self.Wv(source_input)
        else:
            value = source_input

        num_heads = tf.cast(self.num_heads, tf.int32)
        out_channels = tf.cast(self.out_channels, tf.int32)

        query = tf.reshape(query, [-1, num_heads, out_channels])
        key = tf.reshape(key, [-1, num_heads, out_channels])
        if self.use_weight:
            value = tf.reshape(value, [-1, num_heads, out_channels])
        else:
            value = tf.reshape(value, [-1, 1, out_channels])

        attention_output = full_attention_conv_tf(query, key, value)
        final_output = attention_output
        final_output = tf.reduce_mean(final_output, axis=1)

        return final_output


class TransConv(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, num_heads, alpha, dropout, use_bn, use_residual, use_weight, use_act):
        super(TransConv, self).__init__()
        # self.convs = [TransConvLayer(hidden_channels, num_heads, use_weight=use_weight) for _ in
        #               range(num_layers)]
        current_channels = hidden_channels
        self.convs = []
        for _ in range(1):
            self.convs.append(TransConvLayer(current_channels, num_heads, use_weight=use_weight))
            current_channels = current_channels // 2
        self.fcs = [layers.Dense(hidden_channels, kernel_initializer=init)]
        self.bns = [layers.LayerNormalization(axis=-1) for _ in range(1 + 1)]

        self.dropout_layer = layers.Dropout(dropout)
        # self.dropout = dropout
        self.activation = tf.keras.activations.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def call(self, data):

        x = data
        layer_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        # x = self.dropout_layer(x)

        layer_.append(x)

        for i in range(len(self.convs)):

            x = self.convs[i](x, x)

            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            # x = self.dropout_layer(x)
            layer_.append(x)

        # x = self.dense_layer(x)
        return x


class ProcessSingleSampleLayer(tf.keras.layers.Layer):
    def __init__(self, trans_conv, gnn, current_dim):
        super(ProcessSingleSampleLayer, self).__init__()
        self.trans_conv = trans_conv
        self.gnn = gnn
        self.fcs = layers.Dense(current_dim, kernel_initializer=init)
        self.bn = layers.LayerNormalization(axis=-1)
        self.activation = tf.keras.activations.relu

    def call(self, inputs):
        def process_single_sample(sample):
            x1 = self.trans_conv(sample)
            x2 = self.gnn(sample)
            # x = x2 + x1
            x = tf.concat([x1, x2], axis=-1)
            # x = self.fcs(x)
            # x = self.bn(x)
            # x = self.activation(x)
            return x

        # return tf.map_fn(process_single_sample, inputs)
        return tf.vectorized_map(process_single_sample, inputs)

