import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from SGFormer_layer import GCN, TransConv, ProcessSingleSampleLayer
import random
# import tensorflow_addons as tfa

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
init = tf.keras.initializers.GlorotUniform(seed=42)
init1 = tf.keras.initializers.GlorotUniform(seed=42)


class GAEDecoder(tf.keras.Model):
    def __init__(self):
        super(GAEDecoder, self).__init__()

    def call(self, z):
        z_norm = tf.norm(z, axis=2, keepdims=True)
        z_normalized = z / (z_norm + 1e-8)
        adjacency_reconstructed = tf.matmul(z_normalized, z_normalized, transpose_b=True)
        return adjacency_reconstructed


def sampling(args):
    z_mean, z_log_var = args
    return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(shape=K.shape(z_mean))


def get_FC_CAE(batch_size,
               in_channels,
               hidden_channels,
               out_channels,
               num_layers=1,
               num_heads=1,
               alpha=0.5,
               dropout=0,
               use_bn=True,
               use_residual=True,
               use_weight=True,
               use_act=True):

    tg_Feature_input = Input(shape=(in_channels, in_channels), name='tg_data')
    tg_A_input = Input(shape=(in_channels, in_channels), name='tg_A')
    bg_Feature_input = Input(shape=(in_channels, in_channels), name='bg_data')
    bg_A_input = Input(shape=(in_channels, in_channels), name='bg_A')
    mean_matrix = Input(shape=(in_channels, in_channels), name='mean_matrix')

    z_gnn = GCN(32, num_layers, dropout, use_bn, use_act)
    s_gnn = GCN(32, num_layers, dropout, use_bn, use_act)
    z_trans_conv = TransConv(96, num_heads, alpha, dropout, use_bn, use_residual, use_weight, use_act)
    s_trans_conv = TransConv(96, num_heads, alpha, dropout, use_bn, use_residual, use_weight, use_act)

    z_process_layer = ProcessSingleSampleLayer(z_trans_conv, z_gnn, 8)
    s_process_layer = ProcessSingleSampleLayer(s_trans_conv, s_gnn, 8)

    z_mean_layer = Dense(out_channels, kernel_initializer=init, name='z_mean')
    z_log_var_layer = Dense(out_channels, kernel_initializer=init, name='z_log_var')
    z_layer = Lambda(sampling, output_shape=(out_channels,), name='z')
    s_mean_layer = Dense(out_channels, kernel_initializer=init, name='s_mean')
    s_log_var_layer = Dense(out_channels, kernel_initializer=init, name='s_log_var')
    s_layer = Lambda(sampling, output_shape=(out_channels,), name='s')

    z_dense = Dense(32, activation='relu', kernel_initializer=init)
    s_dense = Dense(32, activation='relu', kernel_initializer=init)

    def z_encoder_func(Feature_input):

        z_x = z_process_layer(Feature_input)
        z_x = z_dense(z_x)
        z_mean = z_mean_layer(z_x)
        z_log_var = z_log_var_layer(z_x)

        z = z_layer([z_mean, z_log_var])

        return z_mean, z_log_var, z

    tg_z_mean, tg_z_log_var, tg_z = z_encoder_func(tg_Feature_input)

    def s_encoder_func(Feature_input):

        s_x = s_process_layer(Feature_input)
        s_x = s_dense(s_x)
        s_mean = s_mean_layer(s_x)
        s_log_var = s_log_var_layer(s_x)

        s = s_layer([s_mean, s_log_var])

        return s_mean, s_log_var, s

    tg_s_mean, tg_s_log_var, tg_s = s_encoder_func(tg_Feature_input)
    bg_z_mean, bg_z_log_var, bg_z = z_encoder_func(bg_Feature_input)

    z_encoder = Model([tg_Feature_input], tg_z, name='z_encoder')
    s_encoder = Model([tg_Feature_input], tg_s, name='s_encoder')

    decoder = GAEDecoder()

    # bg_s_zero = tf.zeros_like(tg_s)
    # tg_outputs = decoder(tf.keras.layers.concatenate([tg_s, tg_z], axis=-1))
    # bg_outputs = decoder(tf.keras.layers.concatenate([bg_s_zero, bg_z], axis=-1))

    tg_outputs = decoder(tg_s + tg_z)
    bg_outputs = decoder(bg_z)

    discriminator = Dense(1, kernel_initializer=init1, activation='sigmoid')

    z1 = Lambda(lambda x: x[:int(batch_size / 2), :])(tg_z)
    z2 = Lambda(lambda x: x[int(batch_size / 2):, :])(tg_z)
    s1 = Lambda(lambda x: x[:int(batch_size / 2), :])(tg_s)
    s2 = Lambda(lambda x: x[int(batch_size / 2):, :])(tg_s)

    q_bar = tf.keras.layers.concatenate(
                [tf.keras.layers.concatenate([s1, z2], axis=-1),
                 tf.keras.layers.concatenate([s2, z1], axis=-1)],
                axis=0)

    q = tf.keras.layers.concatenate(
        [tf.keras.layers.concatenate([s1, z1], axis=-1),
         tf.keras.layers.concatenate([s2, z2], axis=-1)],
        axis=0)

    # q_bar = tf.keras.layers.concatenate([s1+z2, s2+z1], axis=0)
    # q = tf.keras.layers.concatenate([s1+z1, s2+z2], axis=0)

    q_bar_score = (discriminator(q_bar) + .1) * .9
    q_score = (discriminator(q) + .1) * .9
    tc_loss = K.log(q_score / (1 - q_score))
    discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)

    DX_loss = tf.keras.losses.mse(K.flatten(tg_A_input), K.flatten(tg_outputs))
    TD_loss = tf.keras.losses.mse(K.flatten(bg_A_input), K.flatten(bg_outputs))
    DX_loss *= in_channels * in_channels
    TD_loss *= in_channels * in_channels

    kl_tg_z = 1 + tg_z_log_var - tf.square(tg_z_mean) - tf.exp(tg_z_log_var)
    kl_tg_z = tf.reduce_sum(kl_tg_z, axis=-1, keepdims=True)
    kl_tg_z *= -0.5

    kl_tg_s = 1 + tg_s_log_var - tf.square(tg_s_mean) - tf.exp(tg_s_log_var)
    kl_tg_s = tf.reduce_sum(kl_tg_s, axis=-1, keepdims=True)
    kl_tg_s *= -0.5

    kl_bg_z = 1 + bg_z_log_var - tf.square(bg_z_mean) - tf.exp(bg_z_log_var)
    kl_bg_z = tf.reduce_sum(kl_bg_z, axis=-1, keepdims=True)
    kl_bg_z *= -0.5

    kl_loss = kl_bg_z + kl_tg_z + kl_tg_s

    mean_dx_loss = K.mean(DX_loss)
    mean_td_loss = K.mean(TD_loss)
    mean_kl_tg_z = K.mean(kl_tg_z)
    mean_kl_tg_s = K.mean(kl_tg_s)
    mean_kl_bg_z = K.mean(kl_bg_z)
    mean_kl_total = K.mean(kl_loss)
    mean_tc_loss = K.mean(tc_loss)
    mean_disc_loss = K.mean(discriminator_loss)

    cae_loss = mean_dx_loss + mean_td_loss + alpha*mean_kl_total + beta*mean_tc_loss + mean_disc_loss

    cae = tf.keras.models.Model(
        inputs=[tg_Feature_input, tg_A_input, bg_Feature_input, bg_A_input, mean_matrix],
        outputs=[tg_outputs, bg_outputs],
        name='contrastive_ae'
    )

    cae.add_loss(cae_loss)

    cae.add_metric(mean_dx_loss, name='dx_mse_loss')
    cae.add_metric(mean_td_loss, name='td_mse_loss')
    cae.add_metric(mean_kl_tg_z, name='kl_tg_z_loss')
    cae.add_metric(mean_kl_tg_s, name='kl_tg_s_loss')
    cae.add_metric(mean_kl_bg_z, name='kl_bg_z_loss')
    cae.add_metric(mean_kl_total, name='kl_total_loss')
    cae.add_metric(mean_tc_loss, name='tc_loss')
    cae.add_metric(mean_disc_loss, name='discriminator_loss')

    opt = tf.keras.optimizers.Adam(learning_rate=0.001,
                                   beta_1=0.9,
                                   beta_2=0.999,
                                   epsilon=1e-07,
                                   amsgrad=False,
                                   name='Adam')

    cae.compile(optimizer=opt)

    return cae, z_encoder, s_encoder, decoder