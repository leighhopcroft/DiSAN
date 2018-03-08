import keras.backend as K
from keras.layers import Layer, Activation, Dropout, BatchNormalization
from disan_keras import exp_mask_for_high_rank, mask_for_high_rank, softmax, meshgrid, scaled_tanh

class MultiDimAttn(Layer):
    def __init__(self, keep_prob=1.0, activation='elu', **kwargs):
        self.keep_prob = keep_prob
        self.activation = activation
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

        map1 = K.dot(rep_tensor, self.map1_kernel) + self.map1_bias
        map1 = Activation(self.activation)(map1)
        map1 = Dropout(1-self.keep_prob)(map1)

        map2 = K.dot(map1, self.map2_kernel) + self.map2_bias
        map2 = Activation(self.activation)(map2)
        map2 = Dropout(1 - self.keep_prob)(map2)

        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = softmax(map2_masked, axis=1)  # bs,sl,vec
        attn_output = K.sum(soft * map2_masked, axis=1) # bs, vec

        return attn_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]) # bs, vec

class DirectionalAttn(Layer):
    def __init__(self, direction=None, keep_prob=1.0, activation='elu', **kwargs):
        self.direction = direction
        self.keep_prob = keep_prob
        self.activation = activation
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
        #rep_tensor = K.print_tensor(rep_tensor, 'rep_tensor')
        #rep_mask = K.print_tensor(rep_mask, 'rep_mask')

        _, isl, ivec = K.int_shape(rep_tensor)
        bs = K.shape(rep_tensor)[0]

        # mask generation
        sl_indices = K.arange(isl)
        #sl_indices = K.print_tensor(sl_indices, 'sl_indices')
        sl_col, sl_row = meshgrid(sl_indices, sl_indices)
        #sl_col = K.print_tensor(sl_col, 'sl_col')
        #sl_row = K.print_tensor(sl_row, 'sl_row')
        if self.direction is None:
            diag = K.eye(isl, dtype='int32') * -K.ones([isl], dtype='int32')
            direct_mask = K.cast(diag + 1, 'bool')
        else:
            if self.direction == 'forward':
                direct_mask = K.greater(sl_row, sl_col)
            else:
                direct_mask = K.greater(sl_col, sl_row)
        #direct_mask = K.print_tensor(direct_mask, 'direct_mask')
        direct_mask_tile = K.tile(K.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        #direct_mask_tile = K.print_tensor(direct_mask_tile, 'direct_mask_tile')
        rep_mask_tile = K.tile(K.expand_dims(rep_mask, 1), [1, isl, 1])  # bs,sl,sl
        #rep_mask_tile = K.print_tensor(rep_mask_tile, 'rep_mask_tile')
        attn_mask = K.all(K.concatenate([K.expand_dims(direct_mask_tile, 0), K.expand_dims(rep_mask_tile, 0)], axis=0),
                          axis=0)  # bs,sl,sl
        #attn_mask = K.print_tensor(attn_mask, 'attn_mask')

        # non-linear
        rep_map = K.dot(rep_tensor, self.rep_map_kernel) + self.rep_map_bias
        #self.rep_map_kernel = K.print_tensor(self.rep_map_kernel, 'rep_map_kernel')
        #self.rep_map_bias = K.print_tensor(self.rep_map_bias, 'rep_map_bias')
        #rep_map = K.print_tensor(rep_map, 'rep_map')
        rep_map = BatchNormalization()(rep_map)
        #rep_map = K.print_tensor(rep_map, 'rep_map_bn')
        rep_map = Activation(self.activation)(rep_map)
        #rep_map = K.print_tensor(rep_map, 'rep_map_act')
        rep_map_tile = K.tile(K.expand_dims(rep_map, 1), [1, isl, 1, 1])  # bs,sl,sl,vec
        #rep_map_tile = K.print_tensor(rep_map_tile, 'rep_map_tile')
        rep_map_dp = Dropout(1 - self.keep_prob)(rep_map)
        #rep_map_dp = K.print_tensor(rep_map_dp, 'rep_map_dp')

        # attention
        dependent = K.dot(rep_map_dp, self.dependent_kernel)
        #self.dependent_kernel = K.print_tensor(self.dependent_kernel, 'dependent_kernel')
        #dependent = K.print_tensor(dependent, 'dependent')
        head = K.dot(rep_map_dp, self.head_kernel)
        #self.head_kernel = K.print_tensor(self.head_kernel, 'head_kernel')
        #head = K.print_tensor(head, 'head')

        dependent_etd = K.expand_dims(dependent, 1)  # bs,1,sl,vec
        #dependent_etd = K.print_tensor(dependent_etd, 'dependent_etd')
        head_etd = K.expand_dims(head, 2)  # bs,sl,1,vec
        #head_etd = K.print_tensor(head_etd, 'head_etd')

        logits = scaled_tanh(dependent_etd + head_etd + self.attn_bias, 5.0)  # bs,sl,sl,vec
        #self.attn_bias = K.print_tensor(self.attn_bias, 'attn_bias')
        #logits = K.print_tensor(logits, 'logits')

        logits_masked = exp_mask_for_high_rank(logits, attn_mask)
        #logits_masked = K.print_tensor(logits_masked, 'logits_masked')
        attn_score = softmax(logits_masked, 2)  # bs,sl,sl,vec
        #attn_score = K.print_tensor(attn_score, 'attn_score')
        attn_score = mask_for_high_rank(attn_score, attn_mask)
        #attn_score = K.print_tensor(attn_score, 'attn_score_masked')

        attn_result = K.sum(attn_score * rep_map_tile, 2)  # bs,sl,vec
        #attn_result = K.print_tensor(attn_result, 'attn_result')

        attn_linear = K.dot(attn_result, self.f_attn_kernel)
        #self.f_attn_kernel = K.print_tensor(self.f_attn_kernel, 'f_attn_kernel')
        #attn_linear = K.print_tensor(attn_linear, 'attn_linear')
        rep_linear = K.dot(rep_map, self.f_rep_kernel)
        #self.f_rep_kernel = K.print_tensor(self.f_rep_kernel, 'f_rep_kernel')
        #rep_linear = K.print_tensor(rep_linear, 'rep_linear')
        fusion_gate = Activation('sigmoid')(attn_linear + rep_linear + self.f_bias)
        #self.f_bias = K.print_tensor(self.f_bias, 'f_bias')
        #fusion_gate = K.print_tensor(fusion_gate, 'fusion_gate')

        output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
        #output = K.print_tensor(output, 'output')
        output = mask_for_high_rank(output, rep_mask)
        #output = K.print_tensor(output, 'output_masked')

        return output

    def compute_output_shape(self, input_shape):
        return input_shape # bs, sl, vec

if __name__ == '__main__':
    from keras.layers import Input, Concatenate, Dense, Reshape, Lambda
    from keras.models import Model
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    b, t, n = 5, 10, 3

    ts_input = Input(shape=(t, ))
    ts_input_pr = Lambda(K.print_tensor, arguments={'message': 'ts_input'})(ts_input)
    ts = Reshape((t, 1))(ts_input_pr)
    ts = Lambda(K.print_tensor, arguments={'message': 'ts'})(ts)

    rep_tensor = Dense(n, activation='tanh')(ts)
    #rep_mask = Input(shape=(t,), dtype='bool')
    rep_tensor = Lambda(K.print_tensor, arguments={'message': 'rep_tensor'})(rep_tensor)

    f_attn = DirectionalAttn(direction='forward')(rep_tensor)
    f_attn = Lambda(K.print_tensor, arguments={'message': 'f_attn'})(f_attn)
    b_attn = DirectionalAttn(direction='backward')(rep_tensor)
    b_attn = Lambda(K.print_tensor, arguments={'message': 'b_attn'})(b_attn)

    rep_3d = Concatenate(axis=-1)([f_attn, b_attn])
    rep_3d = Lambda(K.print_tensor, arguments={'message': 'f_b_concat'})(rep_3d)

    rep_2d = MultiDimAttn()(rep_3d)
    rep_2d = Lambda(K.print_tensor, arguments={'message': 'rep_2d'})(rep_2d)

    output = Dense(1, activation='linear')(rep_2d)
    output = Lambda(K.print_tensor, arguments={'message': 'output'})(output)

    model = Model(ts_input, output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    ts = np.arange(1000).astype(np.float32)
    ts = MinMaxScaler().fit_transform(ts.reshape(-1, 1))[:, 0]
    print(ts.min(), ts.max(), ts.shape)

    X, y = [], []
    for i in range(len(ts)):
        if i >= t:
            X.append(ts[i-t:i])
            y.append(ts[i])

    X, y = np.asarray(X), np.asarray(y)

    print(X.shape, y.shape)

    model.fit(X, y, validation_split=0.1, epochs=1, verbose=0)

