import keras.backend as K
from keras.layers import Layer, Activation, Dropout, BatchNormalization, Dense, Concatenate
from disan_keras import exp_mask_for_high_rank, mask_for_high_rank, softmax, meshgrid, scaled_tanh, get_attn, plot_attn
from collections import OrderedDict
import tensorflow as tf

def tf_print(x, message=''):
    message += ':{}\t'.format(K.int_shape(x))
    tf_print.summarize = 3 if not tf_print.summarize else tf_print.summarize
    return tf.Print(x, [x], message, summarize=tf_print.summarize)


class MultiDimAttn(Layer):
    def __init__(self, dropout=0.0, activation='elu', batch_norm=False, **kwargs):
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.attn_dict = OrderedDict()
        super(MultiDimAttn, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[-1]

        self.map1_kernel = self.add_weight(shape=(input_dim, input_dim),
                                           initializer='glorot_uniform',
                                           name='map1_kernel',
                                           regularizer=None,
                                           constraint=None)

        self.map2_kernel = self.add_weight(shape=(input_dim, input_dim),
                                           initializer='glorot_uniform',
                                           name='map2_kernel',
                                           regularizer=None,
                                           constraint=None)


        self.map1_bias = self.add_weight(shape=(input_dim,),
                                        initializer='zeros',
                                        name='map1_bias',
                                        regularizer=None,
                                        constraint=None)

        self.map2_bias = self.add_weight(shape=(input_dim,),
                                         initializer='zeros',
                                         name='map2_bias',
                                         regularizer=None,
                                         constraint=None)

        self.built = True

    def call(self, inputs, rep_mask=None):
        rep_mask = K.cast(K.ones((K.shape(inputs)[0], K.int_shape(inputs)[1])), dtype='bool') if rep_mask is None else rep_mask
        rep_tensor = inputs

        map1 = K.dot(rep_tensor, self.map1_kernel)
        map1 = K.bias_add(map1, self.map1_bias)
        if self.batch_norm:
            map1 = BatchNormalization()(map1)
        map1 = Activation(self.activation)(map1)
        map1 = Dropout(self.dropout)(map1)

        map2 = K.dot(map1, self.map2_kernel)
        map2 = K.bias_add(map1, self.map2_bias)
        if self.batch_norm:
            map2 = BatchNormalization()(map2)
        map2 = Activation(self.activation)(map2)
        map2 = Dropout(self.dropout)(map2)

        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = softmax(map2_masked, axis=1)  # bs,sl,vec
        self.attn_dict['multidim_attn'] = soft
        attn_output = K.sum(soft * map2_masked, axis=1) # bs, vec

        return attn_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]) # bs, vec


class DirectionalAttn(Layer):
    def __init__(self, direction=None, dropout=0.0, activation='elu', batch_norm=False, **kwargs):
        self.direction = direction
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.attn_dict = OrderedDict()
        super(DirectionalAttn, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[-1]

        self.rep_map_kernel = self.add_weight(shape=(input_dim, input_dim),
                                              initializer='glorot_uniform',
                                              name='rep_map_kernel',
                                              regularizer=None,
                                              constraint=None,
                                              dtype=K.floatx())

        self.rep_map_bias = self.add_weight(shape=(input_dim,),
                                              initializer='zeros',
                                              name='rep_map_bias',
                                              regularizer=None,
                                              constraint=None)

        self.head_kernel = self.add_weight(shape=(input_dim, input_dim),
                                              initializer='glorot_uniform',
                                              name='head_kernel',
                                              regularizer=None,
                                              constraint=None)

        self.dependent_kernel = self.add_weight(shape=(input_dim, input_dim),
                                           initializer='glorot_uniform',
                                           name='dependent_kernel',
                                           regularizer=None,
                                           constraint=None)

        self.attn_bias = self.add_weight(shape=(input_dim,),
                                            initializer='zeros',
                                            name='attn_bias',
                                            regularizer=None,
                                            constraint=None)

        self.f_rep_kernel = self.add_weight(shape=(input_dim, input_dim),
                                              initializer='glorot_uniform',
                                              name='f_rep_kernel',
                                              regularizer=None,
                                              constraint=None)

        self.f_attn_kernel = self.add_weight(shape=(input_dim, input_dim),
                                            initializer='glorot_uniform',
                                            name='f_attn_kernel',
                                            regularizer=None,
                                            constraint=None)

        self.f_bias = self.add_weight(shape=(input_dim,),
                                            initializer='zeros',
                                            name='f_bias',
                                            regularizer=None,
                                            constraint=None)

        self.built = True

    def call(self, inputs, rep_mask=None):
        rep_mask = K.cast(K.ones((K.shape(inputs)[0], K.int_shape(inputs)[1])),
                          dtype='bool') if rep_mask is None else rep_mask

        rep_tensor = inputs

        _, isl, ivec = K.int_shape(rep_tensor)
        bs = K.shape(rep_tensor)[0]

        # mask generation
        sl_indices = K.arange(isl)
        sl_col, sl_row = meshgrid(sl_indices, sl_indices)
        if self.direction is None:
            diag = K.eye(isl, dtype='int32') * -K.ones([isl], dtype='int32')
            direct_mask = K.cast(diag + 1, 'bool')
        else:
            if self.direction == 'forward':
                direct_mask = K.greater(sl_row, sl_col)
            else:
                direct_mask = K.greater(sl_col, sl_row)
        direct_mask_tile = K.tile(K.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = K.tile(K.expand_dims(rep_mask, 1), [1, isl, 1])  # bs,sl,sl
        attn_mask = K.all(K.concatenate([K.expand_dims(direct_mask_tile, 0), K.expand_dims(rep_mask_tile, 0)], axis=0),
                          axis=0)  # bs,sl,sl

        # non-linear
        rep_map = K.dot(rep_tensor, self.rep_map_kernel)
        rep_map = K.bias_add(rep_map, self.rep_map_bias)
        if self.batch_norm:
            rep_map = BatchNormalization()(rep_map)
        rep_map = Activation(self.activation)(rep_map)
        rep_map_tile = K.tile(K.expand_dims(rep_map, 1), [1, isl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = Dropout(self.dropout)(rep_map)

        # attention
        dependent = K.dot(rep_map_dp, self.dependent_kernel)
        head = K.dot(rep_map_dp, self.head_kernel)

        dependent_etd = K.expand_dims(dependent, 1)  # bs,1,sl,vec
        head_etd = K.expand_dims(head, 2)  # bs,sl,1,vec
        logits_pre_act = dependent_etd + head_etd
        logits_pre_act = K.bias_add(logits_pre_act, self.attn_bias)
        if self.batch_norm:
            logits_pre_act = BatchNormalization()(logits_pre_act)
        logits = scaled_tanh(logits_pre_act, 5.0)  # bs,sl,sl,vec

        logits_masked = exp_mask_for_high_rank(logits, attn_mask)
        attn_score = softmax(logits_masked, 2)  # bs,sl,sl,vec
        attn_score = mask_for_high_rank(attn_score, attn_mask)
        self.attn_dict['{dir}_attn_full'.format(dir=self.direction)] = attn_score

        attn_result = K.sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        attn_linear = K.dot(attn_result, self.f_attn_kernel)
        rep_linear = K.dot(rep_map, self.f_rep_kernel)
        fusion_pre_act = attn_linear + rep_linear
        fusion_pre_act = K.bias_add(fusion_pre_act, self.f_bias)
        if self.batch_norm:
            fusion_pre_act = BatchNormalization()(fusion_pre_act)
        fusion_gate = Activation('sigmoid')(fusion_pre_act)
        self.attn_dict['{dir}_attn_fusion'.format(dir=self.direction)] = fusion_gate

        output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
        output = mask_for_high_rank(output, rep_mask)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape  # bs, sl, vec


class DISAN(Layer):
    def __init__(self, hidden_dim=None, dropout=0.0, activation='elu', batch_norm=False, return_attn=False, **kwargs):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.attn_dict = OrderedDict()
        if hidden_dim:
            self.output_dim = hidden_dim * 2
        else:
            self.output_dim = None
        super(DISAN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DISAN, self).build(input_shape)

    def call(self, inputs, rep_mask=None):
        rep_mask = K.cast(K.ones((K.shape(inputs)[0], K.int_shape(inputs)[1])),
                          dtype='bool') if rep_mask is None else rep_mask

        if self.hidden_dim:
            rep_tensor = Dense(self.hidden_dim, activation=self.activation)(inputs)
            if self.batch_norm:
                rep_tensor = BatchNormalization()(rep_tensor)
            rep_tensor = Dropout(self.dropout)(rep_tensor)
        else:
            rep_tensor = inputs
            self.output_dim = K.int_shape(inputs)[-1] * 2

        f_attn_layer = DirectionalAttn(direction='forward', dropout=self.dropout,
                                       activation=self.activation, batch_norm=self.batch_norm)
        f_attn = f_attn_layer(rep_tensor, rep_mask=rep_mask)
        b_attn_layer = DirectionalAttn(direction='backward', dropout=self.dropout,
                                       activation=self.activation, batch_norm=self.batch_norm)
        b_attn = b_attn_layer(rep_tensor, rep_mask=rep_mask)


        rep_3d = Concatenate(axis=-1)([f_attn, b_attn])

        m_attn_layer = MultiDimAttn(dropout=self.dropout,
                                    activation=self.activation,
                                    batch_norm=self.batch_norm)
        rep_2d = m_attn_layer(rep_3d, rep_mask=rep_mask)

        # merge dicts
        for d in [f_attn_layer.attn_dict, b_attn_layer.attn_dict, m_attn_layer.attn_dict]:
            for k, v in d.items():
                self.attn_dict[k] = v

        return rep_2d

    def compute_output_shape(self, input_shape):
        if not self.output_dim:
            self.output_dim = input_shape[-1]*2
        return (input_shape[0], self.output_dim)

if __name__ == '__main__':
    from keras.layers import Input, Reshape
    from keras.models import Model
    from keras.optimizers import Adadelta
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
    import numpy as np
    import os
    from sklearn.preprocessing import MinMaxScaler

    b, t, n = 5, 10, 64

    tf_print.summarize = 100

    ts_input = Input(shape=(t, ))
    ts = Reshape((t, 1))(ts_input)

    disan = DISAN(n, dropout=0.2)
    rep_2d = disan(ts)

    output = Dense(1, activation='linear')(rep_2d)

    model = Model(ts_input, output)
    opt = Adadelta(lr=0.5)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    model.summary()

    ts = np.arange(10000).astype(np.float32)
    ts = MinMaxScaler().fit_transform(ts.reshape(-1, 1))[:, 0]
    print(ts.min(), ts.max(), ts.shape)

    X, y = [], []
    for i in range(len(ts)):
        if i >= t:
            X.append(ts[i-t:i])
            y.append(ts[i])

    X, y = np.asarray(X), np.asarray(y)
    train_len = int(len(X) * 0.9)
    X_train, y_train = X[:train_len, ...], y[:train_len, ...]
    X_test, y_test = X[train_len:, ...], y[train_len:, ...]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    filepath = 'models'
    callbacks = [
        ReduceLROnPlateau(patience=0, verbose=1),
        EarlyStopping(patience=3, verbose=1),
        #ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(os.path.join(filepath, 'logs'), histogram_freq=1)
    ]

    model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=1, callbacks=callbacks)
    print(model.evaluate(X_test, y_test, batch_size=len(X_test)))

    attn = get_attn(X_test[0, ...], model, disan.attn_dict)
    for k, v in attn.items():
        print(k, v.shape)

    plot_attn(attn)



