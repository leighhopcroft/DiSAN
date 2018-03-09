import keras.backend as K
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Lambda, Multiply, Masking
import matplotlib.pyplot as plt
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def disan(rep_tensor, rep_mask, keep_prob=1., activation='elu'):
    fw_res = directional_attention_with_dense(
        rep_tensor, rep_mask, 'forward',
        keep_prob, activation)
    bw_res = directional_attention_with_dense(
        rep_tensor, rep_mask, 'backward',
        keep_prob, activation)

    seq_rep = K.concatenate([fw_res, bw_res], -1)

    sent_rep = multi_dimensional_attention(seq_rep, rep_mask, keep_prob, activation)
    return sent_rep

def directional_attention_with_dense(rep_tensor, rep_mask, direction=None, keep_prob=1., activation='elu'):
    _, isl, ivec = K.int_shape(rep_tensor)
    bs = K.shape(rep_tensor)[0]

    # mask generation
    sl_indices = K.arange(isl)
    sl_col, sl_row = meshgrid(sl_indices, sl_indices)
    if direction is None:
        diag = K.eye(isl, dtype='int32') * -K.ones([isl], dtype='int32')
        direct_mask = K.cast(diag + 1, 'bool')
    else:
        if direction == 'forward':
            direct_mask = K.greater(sl_row, sl_col)
        else:
            direct_mask = K.greater(sl_col, sl_row)
    direct_mask_tile = K.tile(K.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
    rep_mask_tile = K.tile(K.expand_dims(rep_mask, 1), [1, isl, 1])  # bs,sl,sl
    attn_mask = K.all(K.concatenate([K.expand_dims(direct_mask_tile, 0), K.expand_dims(rep_mask_tile, 0)], axis=0), axis=0)  # bs,sl,sl

    # non-linear
    rep_map = Dense(ivec, use_bias=True, bias_initializer='zeros', name='bn_dense_map', activation=None)(rep_tensor)
    rep_map = BatchNormalization()(rep_map)
    rep_map = Activation(activation)(rep_map)
    rep_map_tile = K.tile(K.expand_dims(rep_map, 1), [1, isl, 1, 1])  # bs,sl,sl,vec
    rep_map_dp = Dropout(1 - keep_prob)(rep_map)

    # attention
    dependent = Dense(ivec, use_bias=False, activation=None)(rep_map_dp)
    head_w_bias = Dense(ivec, use_bias=True, activation=None)(rep_map_dp)
    dependent_etd = K.expand_dims(dependent, 1)  # bs,1,sl,vec
    head_etd = K.expand_dims(head_w_bias, 2)  # bs,sl,1,vec

    logits = scaled_tanh(dependent_etd + head_etd, 5.0)  # bs,sl,sl,vec

    logits_masked = exp_mask_for_high_rank(logits, attn_mask)
    attn_score = softmax(logits_masked, 2)  # bs,sl,sl,vec
    attn_score = mask_for_high_rank(attn_score, attn_mask)

    attn_result = K.sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

    attn_linear = Dense(ivec, activation=None, use_bias=False)(attn_result)
    rep_linear_w_bias = Dense(ivec, activation=None, use_bias=True)(rep_map)
    fusion_gate = Activation('sigmoid')(attn_linear + rep_linear_w_bias)

    output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
    output = mask_for_high_rank(output, rep_mask)

    return output


def multi_dimensional_attention(rep_tensor, rep_mask,
                                keep_prob=1., activation='elu'):

    ivec = K.int_shape(rep_tensor)[2]

    map1 = Dense(ivec, use_bias=True, bias_initializer='zeros', name='dense_map1', activation=activation)(rep_tensor)
    map1 = Dropout(1-keep_prob, name='map1_dropout')(map1)

    map2 = Dense(ivec, use_bias=True, bias_initializer='zeros', name='dense_map2', activation='linear')(map1)
    map2 = Dropout(1 - keep_prob, name='map2_dropout')(map2)

    map2_masked = exp_mask_for_high_rank(map2, rep_mask)

    soft = Lambda(softmax, arguments={'axis': 1}, name='softmax')(map2_masked)  # bs,sl,vec
    attn_output = Multiply(name='attn_mul')([soft, map2])
    attn_output = Lambda(K.sum, arguments={'axis': 1}, name='attn_sum')(attn_output) # bs, vec

    return attn_output

def mask_for_high_rank(val, val_mask):
    val_mask = K.expand_dims(val_mask, -1)
    return val * K.cast(val_mask, K.floatx())

def exp_mask_for_high_rank(val, val_mask):
    val_mask = K.expand_dims(val_mask, -1)
    return val + ((1 - K.cast(val_mask, K.floatx())) * VERY_NEGATIVE_NUMBER)

def softmax(logits, axis=0):
    exp = K.exp(logits)
    # avoid div0 errors by adding a very small constant K.epsilon() to the denominator
    sumexp = K.sum(exp, axis, keepdims=True) + K.epsilon()
    return exp / sumexp

def tanh(x):
    return (2.0/(1.0 + K.exp(-2.0*x)))-1.0

def scaled_tanh(x, scale=5.):
    return scale * tanh(1./scale * x)

def meshgrid(*args):
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(K.reshape(K.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [size(x) for x in args]

    #output_dtype = ops.convert_to_tensor(args[0]).dtype.base_dtype

    if ndim > 1:
        output[0] = K.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = K.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]

    mult_fact = K.ones(shapes, dtype='int32')
    return [x * mult_fact for x in output]

def size(x):
    return np.prod(K.int_shape(x))

def get_attn(input, model, attn_tensors_dict, keys=[]):
    assert len(attn_tensors_dict) > 0
    if len(keys) == 0:
        keys = list(attn_tensors_dict.keys())
    nsamples = len(input.shape)
    if nsamples == 1:
        input = input[np.newaxis, ...]
    attn_func = K.function([model.layers[0].input, K.learning_phase()],
                           [v for k, v in attn_tensors_dict.items() if k in keys])
    attn = attn_func([input, 0])
    if nsamples == 1 or input.shape[0] == 1:
        return {k: v[0] for k, v in zip(keys, attn)}
    else:
        return {k: v for k, v in zip(keys, attn)}


def plot_attn(attn_dict_np, figsize=(8, 6)):
    attn_full = attn_dict_np['forward_attn_full'].mean(axis=-1) + attn_dict_np['backward_attn_full'].mean(axis=-1)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    ax1.imshow(attn_full, cmap=plt.get_cmap('Greys'))
    ax2.imshow(attn_dict_np['forward_attn_fusion'].mean(axis=-1, keepdims=True).T, cmap=plt.get_cmap('Greys'))
    ax3.imshow(attn_dict_np['backward_attn_fusion'].mean(axis=-1, keepdims=True).T, cmap=plt.get_cmap('Greys'))
    ax4.imshow(attn_dict_np['multidim_attn'].mean(axis=-1, keepdims=True).T, cmap=plt.get_cmap('Greys'))
    plt.show()

if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model

    b, t, n = 5, 10, 3

    rep_tensor = Input(shape=(t, n))
    rep_mask = Input(shape=(t, ), dtype='bool')

    out = disan(rep_tensor, rep_mask)

    model = Model([rep_tensor, rep_mask], out)
    model.summary()