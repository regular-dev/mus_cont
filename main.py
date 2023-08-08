import os
import numpy as np
import wave
import random
import pickle
import struct
import sys
from keras.layers import Input, LSTM, concatenate, Add, Dense, Dropout, Conv1DTranspose, Reshape, Flatten, Conv1D, Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf

WAV_VAL_MIN = -10000
WAV_VAL_MAX = 10000

def pad_array(arr, n):
    if len(arr) < n:
        padded_arr = np.pad(arr, (0, n - len(arr)), 'constant')
    else:
        padded_arr = arr
    return padded_arr

def min_max_normalize(arr, min_val, max_val):
    range_val = max_val - min_val
    normalized_arr = (arr - min_val) / range_val
    return normalized_arr

def apply_threshold(arr, min_threshold, max_threshold):
    arr = np.maximum(arr, min_threshold)
    arr = np.minimum(arr, max_threshold)

    return arr

def create_random_array(shape):
    return np.random.rand(shape)

def get_random_array_sequence(_in_vec, _n):
    # Select random starting index, random offset from the audio end
    start_idx = np.random.randint(0, len(_in_vec) - _n - 12)

    _out = _in_vec[start_idx : start_idx + _n]
    _lbl = np.array(_in_vec[start_idx + _n + 1 : start_idx + _n + 1 + 10]).flatten() # 1 sec label

    if start_idx + _n + 1 >= len(_in_vec):
        print("Error: Invalid index")
        exit(0)

    # lbl len - 44100

    return (_out, _lbl)

def shuffle_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length")

    pairs = list(zip(list1, list2))
    random.shuffle(pairs)
    shuffled_list1, shuffled_list2 = zip(*pairs)

    return shuffled_list1, shuffled_list2

def list_dir_files(_dir_path):
    file_list = []
    
    for root, _, files in os.walk(_dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    
    return file_list

def save_as_wav(arrays, filename):
    mix = np.int16(np.hstack(arrays))

    with wave.open(filename, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(44100)
        file.writeframes(mix.tobytes())

    print(f"File saved as {filename}")

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        _displayed = False

        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        duration = num_frames / float(frame_rate)

        print("sample width : {}".format(sample_width))

        audio_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)
        audio_data = audio_data.reshape(-1, num_channels)

        # convert to 0.1 chunks
        chunk_size = int(frame_rate * 0.1)
        num_chunks = int(np.ceil(num_frames / chunk_size))
        serialized_audio = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, num_frames)
            chunk = audio_data[start:end, :]

            chunk = pad_array(chunk, chunk_size)
            chunk = apply_threshold(chunk, WAV_VAL_MIN, WAV_VAL_MAX)
            chunk = min_max_normalize(chunk, WAV_VAL_MIN, WAV_VAL_MAX)

            if _displayed == False:
                print("chunk : {}".format(chunk))
                print("chunk flatten : {}".format(len(chunk.flatten())))
                print("chunk min : {} | chunk max : {}".format(chunk.flatten().min(), chunk.flatten().max()))
                _displayed = True

            if len(chunk.flatten()) > 5000:
                print("File : {} | {} | {}".format(file_path, i, len(chunk.flatten())))


            serialized_audio.append(chunk.flatten())

        return serialized_audio, duration

def create_ds(data_folder):
    _ds = []
    file_list = list_dir_files(data_folder)

    # Loop over each file and serialize into numpy arrays
    for (f_idx, file_path) in enumerate(file_list):
        serialized_audio, duration = read_wav_file(file_path)
        print("{}: {} seconds, {} chunks".format(file_path, duration, len(serialized_audio)))

        for _ in range(200):
            entry = get_random_array_sequence(serialized_audio, random.randint(20, 80))
            _ds.append(entry)


    with open('dataset.pkl', 'wb') as f:
        pickle.dump(_ds, f)

def create_resnet_rnn():
    x = Input(shape=(None, 4410))

    lstm_1 = LSTM(256, return_sequences=True, dropout=0.1)(x)
    lstm_2 = LSTM(128, return_sequences=True, dropout=0.1)(lstm_1)
    lstm_3 = LSTM(256, return_sequences=True, dropout=0.1)(lstm_2)

    merged_layer = Concatenate(axis=-1)([lstm_2, lstm_3])

    lstm_4 = LSTM(441, return_sequences=False)(merged_layer) # 44100

    rs_1 = Reshape((1, 441))(lstm_4)

    ct_1 = Conv1DTranspose(441, kernel_size=10, strides=10, padding='same', activation='relu')(rs_1)
    ct_1 = Conv1DTranspose(441, kernel_size=10, strides=10, padding='same', activation='sigmoid')(ct_1)

    f = Flatten()(ct_1)

    model = Model(inputs=x, outputs=f)
    model.summary()

    return model

def train_net(ds_path, mdl_state):
    print("Keras backend : {}".format(K.backend()))

    model = create_resnet_rnn()

    if mdl_state != "":
        model = load_model(mdl_state)

    opt = Adam(learning_rate=3e-4)
    model.compile(optimizer=opt)

    epochs = 300
    batch_size = 8

    loss_fn = tf.keras.losses.MeanSquaredError()

    collect_x_arr = []
    collect_y_arr = []

    print("Epoch : {} | batch_size : {}", epochs, batch_size)

    with open(ds_path, 'rb') as f:
        ds = pickle.load(f)
        overall_idx = 0

        for e in range(epochs):
            epoch_loss = 0.0
            batches_num = 0

            random.shuffle(ds)

            for (_in, _lbl) in ds:

                _in = np.array([_in])
                _lbl = np.array([_lbl])

                collect_x_arr.append(_in)
                collect_y_arr.append(_lbl)

                if len(collect_x_arr) == batch_size:
                    current_loss = 0.0
                    with tf.GradientTape() as tape:
                        for b_i in range(len(collect_x_arr)):
                            logits = model(collect_x_arr[b_i], training=True)
                            current_loss += loss_fn(y_true=collect_y_arr[b_i], y_pred=logits)
                            model.reset_states()

                        grad = tape.gradient(current_loss / batch_size, model.trainable_weights)
                        opt.apply_gradients(zip(grad, model.trainable_weights))

                    collect_x_arr = []
                    collect_y_arr = []

                    batch_loss = current_loss / batch_size
                    epoch_loss += batch_loss
                    batches_num += 1

                overall_idx += 1

            if e != 0 and e % 100 == 0:
                model.save("model_snap_{}".format(e))

            print("Epoch {} loss : {:.5f}".format(e, epoch_loss / batches_num))

    model.save("model_{}.state".format(epochs))

# This function tries to continue a random sequence of samples from train dataset
def test_net(ds_path, mdl_state):
    print("keras backend : {}".format(K.backend()))

    # each sample is 0.1 sec
    n_all = 60
    n_prev = 100 # history samples before stack limit

    stack = []
    stack_all = []

    model = load_model(mdl_state)

    with open(ds_path, 'rb') as f:
        ds = pickle.load(f)
        random.shuffle(ds)

        for i in range(len(ds[0][0]) - 10):
            # We can test an output with the random noise as input
            # entry_arr = create_random_array(4410)

            entry_arr = ds[0][0][i]

            stack.append(entry_arr)
            stack_all.append(entry_arr)

    for i in range(n_all):
        if i%50 == 0:
            print("Done : {} iters".format(i))

        out = model(np.array([stack]))
        out = np.array(out).reshape(-1)
        out = np.split(out, 10)

        model.reset_states()

        for j in out:
            stack.append(j)
            stack_all.append(j)

        if len(stack) > n_prev:
            stack.pop(0)


    for i in range(len(stack_all)):
        stack_all[i] = stack_all[i] * 20000 - 10000

    save_as_wav(stack_all, 'out.wav')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage python main.py <subcommand> <args_if_required>")
        exit(0)

    if sys.argv[1] == 'dataset':
        ds_folder = sys.argv[2]
        create_ds(ds_folder)

    if sys.argv[1] == 'train':
        dataset = sys.argv[2]
        mdl_state = ''

        if len(sys.argv) == 4:
            mdl_state = sys.argv[3]

        train_net(dataset, mdl_state)

    if sys.argv[1] == 'test':
        if len(sys.argv) != 4:
            print("Error: invalid number of arguments")
            exit(0)

        dataset = sys.argv[2]
        mdl_state = sys.argv[3]
        test_net(dataset, mdl_state)
