'''Use an autoencoder to learn a dense encoding of classic cartpole from pixels.
Run using
xvfb-run -a -s "-screen 99 1400x900x24" python model_based_learning.py train

Experiment log
* Start with https://blog.keras.io/building-autoencoders-in-keras.html
* Dense-only auto-encoder had 30M parameters and had trouble getting below 63k MSE
* Conv autoencoder with 4k params got to ~370 MSE. Remaining loss probably due to
  * lack of time dependency
  * unknown action
  * input not unit variance
  * background is 1
* scaling factor 5 made autoencoder incapable of preserving fine detail. loss bottomed at 5e-3
* Cropping slightly to make divisible by 8 so we can use 2x scaling made loss < 1e-4
* Decreasing bottleneck capacity increases loss but visually looks better
* Adding actions tiled with input images performed worse than before bottleneck layer
* Adding dense layers between action input and bottleneck helped imagination perform much better
'''

import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model
from keras.models import load_model

def load_data(window=4):
    env_name = 'cartpole'
    frames = np.load('%s_frames.npy' % env_name)
    rewards = np.load('%s_rewards.npy' % env_name)
    actions = np.load('%s_actions.npy' % env_name)
    observations = np.load('%s_observations.npy' % env_name)

    windowed_frames = np.expand_dims(eps_to_stacked_window(frames, window=window), -1)
    # Verify first 4 frames are equal
    assert np.array_equal(np.stack(frames[0][:window]), np.squeeze(windowed_frames[0]))
    # Verify frames 1-5 are equal
    windowed_frames_next = np.expand_dims(eps_to_stacked_window(frames, offset=True, window=window), -1)
    assert np.array_equal(np.stack(frames[0][1:window+1]), np.squeeze(windowed_frames_next[0]))
    # Frame 1 is frame 0 when offset by 1
    assert np.array_equal(windowed_frames[0,1], windowed_frames_next[0,0])
    if window > 2:
        assert np.array_equal(windowed_frames[0,2], windowed_frames_next[0,1])
    assert windowed_frames_next.shape == windowed_frames.shape

    windowed_actions = eps_to_stacked_window(actions, window=window)

    # Only return the last item in the stack for windowed_frames_next  and windowed_actions since we
    # only want to predict the next frame based on the most recent action.
    return frames, windowed_frames, windowed_frames_next[:, window-1:], windowed_actions[:, window-1:]

def sliding_window(a, window, step_size=1):
    '''
    Input is list of `shape` np arrays of length N
    Output is N - 4 x 4 x `shape`
    '''
    end = a.shape[0]
    #return np.moveaxis(np.stack([a[i:end-window+i+1:step_size] for i in range(window)]), 0, -1)
    # TimeDistributed looks at axis 1
    return np.moveaxis(np.stack([a[i:end-window+i+1:step_size] for i in range(window)]), 0, 1)

def eps_to_stacked_window(a, window, offset=False):
    if offset:
        return np.vstack([sliding_window(np.stack(x)[1:], window=window) for x in a])
    else:
        return np.vstack([sliding_window(np.stack(x)[:-1], window=window) for x in a])

def preprocess(b):
    '''Unit variance, invert, image dims divisible by 8'''
    # TODO: make this more general by zero padding instead of cropping.
    return 1 - b[..., :96, :144, :]  / 255.

def make_model(windowed_frames, encoding_dim=4):
    # TODO: skip connections
    # this is our input placeholder
    input_img = Input(shape=preprocess(windowed_frames[:1]).shape[1:])
    input_action = Input(shape=(2,))
    x = input_img
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
    pre_flat_shape = x.shape[1:].as_list()
    x = Flatten()(x)
    x = Dense(32, activation='selu')(x)
    x2 = Dense(16, activation='selu')(input_action)
    x2 = Dense(16, activation='selu')(x2)
    x2 = Dense(16, activation='selu')(x2)
    x = keras.layers.concatenate([x, x2])
    # Bottleneck here!
    x = Dense(encoding_dim, name='bottleneck', activation='selu')(x)
    # Start scaling back up
    # No frame stack for output
    pre_flat_shape[0] = 1
    x = Dense(np.product(pre_flat_shape), activation='selu')(x)
    x = Reshape(pre_flat_shape)(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    decoded = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)

    # this model maps an input to its reconstruction
    model = Model([input_img, input_action], decoded, name='autoencoder')

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[])
    model.summary()

    return model



def get_batch(windowed_frames, windowed_frames_next, windowed_actions, batch_size, indices):
    while True:
        for i in indices:
            start = i * batch_size
            end = (i + 1) * batch_size
            one_hot_actions = np.zeros((batch_size, 2))
            one_hot_actions[range(batch_size), windowed_actions[start:end]] = 1
            yield (
                [preprocess(windowed_frames[start:end]), one_hot_actions],
                preprocess(windowed_frames_next[start:end])
            )


def train(model, windowed_frames, windowed_frames_next, windowed_actions,
          epochs=10, batch_size=64, validation_split=0.9, shuffle=True, save_model_file='autoencoder.h5'):
    print('Logging to Tensorboard in ./logs')
    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S')))

    num_batches = len(windowed_frames) // batch_size
    if shuffle:
        indices = np.random.permutation(num_batches)
    else:
        indices = range(num_batches)

    model.fit_generator(
        get_batch(
            windowed_frames, windowed_frames_next, windowed_actions,
            batch_size, indices[:int(num_batches * (validation_split))]),
        num_batches * (validation_split),
        epochs=epochs,
        verbose=1,
        callbacks=[tb_callback],
        validation_data=get_batch(
            windowed_frames, windowed_frames_next, windowed_actions,
            batch_size, indices[int(num_batches * (validation_split)):]),
        validation_steps=int(num_batches * (1-validation_split))
    )
    model.save(save_model_file)

def get_encoded_prediction(model):
    encoder_model = Model(model.input, model.get_layer('bottleneck').output, name='encoder')
    encoded_pred = encoder_model.predict([preprocess(windowed_frames[:100]), windowed_actions[:100]])
    print(encoded_pred.shape)
    plt.plot(encoded_pred)
    plt.savefig('encoded2.png')
    # Compare discontinuities to episode lengths
    print([len(x) for x in frames[:10]])

if __name__ == '__main__':
    print('loading data')
    frames, windowed_frames, windowed_frames_next, windowed_actions = load_data(window=3)
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        model = make_model(windowed_frames)
        train(model, windowed_frames, windowed_frames_next, windowed_actions)
    else:
        model = load_model('autoencoder.h5')
        get_encoded_prediction(model)
