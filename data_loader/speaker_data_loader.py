import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from base.base_data_loader import BaseDataLoader

class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
        self.labels_names = []
        self.train = tf.data.Dataset.from_tensor_slices([]) 
        self.validation = tf.data.Dataset.from_tensor_slices([])
        self.test = tf.data.Dataset.from_tensor_slices([])
        self.kaggle_test = tf.data.Dataset.from_tensor_slices([])

    def get_test_labels(self):
        true_labels = np.concatenate([label for spectrograms, label in self.test], axis=0)
        return true_labels

    def squeeze(self, audio, labels=None):
        audio = tf.squeeze(audio, axis=-1)
        if not labels == None:
            return audio, labels
        else:
            return audio
    
    def get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def inspect_spectrgoram(self):
        for audio, labels in self.train.take(1):  
            spectrogram = self.get_spectrogram(audio[1])
            print(f'Label Value................................: {labels[1]}')
            print(f'Label Shape(Batch Size, N_Labels)..............: {labels.shape}')
            print(f'Audio Shape(Batch Size, Max SampleRate)....: {audio.shape}')
            print(f'Inspectrogram Shape........................: {spectrogram.shape}')

    def convert_dataset_to_spectrogram(self, dataset):
        return dataset.map(map_func = lambda audio, label: (self.get_spectrogram(audio), label),
                           num_parallel_calls=tf.data.AUTOTUNE)
    
    def process_dataset(self):
        self.train = self.convert_dataset_to_spectrogram(self.train)
        self.validation = self.convert_dataset_to_spectrogram(self.validation)
        self.test = self.convert_dataset_to_spectrogram(self.test)

    def dataset_snapshot(self):
        print('Train Dataset................(Shapes and Length):', self.train.take(1), self.train.cardinality().numpy())
        print('Validation Dataset...........(Shapes and Length):', self.validation.take(1), self.validation.cardinality().numpy())
        print('Test Dataset.................(Shapes and Length):', self.test.take(1), self.test.cardinality().numpy())
        print('Kaggle Dataset...............(Shapes and Length):', self.kaggle_test.take(1), self.kaggle_test.cardinality().numpy())

    def inspect_dataset(self):
        print('...........Audio Dataset..........')
        print(f'Classes Names.....: \n{self.labels_names}')
        print(f'Train Element Inpect....: \n{self.train.element_spec}')
        print(f'Kaggle Element Inpect....: \n{self.kaggle_test.element_spec}')

    def get_train_dataset(self):
        return self.train.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    
    def get_validation_dataset(self):
        return self.validation.cache().prefetch(tf.data.AUTOTUNE)

    def get_test_dataset(self):
        return self.test.cache().prefetch(tf.data.AUTOTUNE)
    
    def get_shapes(self):
        for spectrogram, label in self.train.take(1):
            return (spectrogram[1].shape, label.shape)

    def load_dataset(self):
        self.train, self.validation = tf.keras.utils.audio_dataset_from_directory(f'{self.config.location.audios}',
                                                                                    label_mode='categorical', 
                                                                                    output_sequence_length=16000,
                                                                                    batch_size=self.config.trainer.batch_size,
                                                                                    validation_split=0.3,
                                                                                    seed=42,
                                                                                    subset='both') 
        
        self.labels_names = np.array(self.train.class_names)

        self.train = self.train.map(self.squeeze, tf.data.AUTOTUNE)
        self.validation= self.validation.map(self.squeeze, tf.data.AUTOTUNE)

        self.test = self.validation.shard(num_shards=2, index=0)
        self.validation = self.validation.shard(num_shards=2, index=1)

    def load_kaggle_test(self):
        test = pd.read_csv(self.config.location.test)
        data = []
        for wav_location in test.file_path:
            raw_audio = tf.io.read_file(f'./data/{wav_location}')
            wave, sr = tf.audio.decode_wav(raw_audio, desired_channels=-1, desired_samples=16000, name=None)
            data.append(wave)

        self.kaggle_test = tf.data.Dataset.from_tensor_slices(data).batch(32)
        self.kaggle_test = self.kaggle_test.map(self.squeeze, tf.data.AUTOTUNE)
        self.kaggle_test = self.kaggle_test.map(self.get_spectrogram, tf.data.AUTOTUNE)

    def get_kaggle_test_dataset(self):
        return self.kaggle_test.cache().prefetch(tf.data.AUTOTUNE)

    def get_kaggle_test_shape(self):
        for spectrogram in self.kaggle_test.take(1):
            return spectrogram[1].shape


