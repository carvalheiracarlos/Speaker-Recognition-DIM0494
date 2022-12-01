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
        self.train_spectrograms = tf.data.Dataset.from_tensor_slices([]) 
        self.validation = tf.data.Dataset.from_tensor_slices([])
        self.validation_spectrograms = tf.data.Dataset.from_tensor_slices([])
        self.test = tf.data.Dataset.from_tensor_slices([])
        self.test_spectrograms = tf.data.Dataset.from_tensor_slices([])

    def squeeze(self, audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
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

    def inspect_dataset(self):
        print('...........Audio Dataset..........')
        print(f'Classes Names.....: \n{self.labels_names}')
        print(f'Element Inpect....: \n{self.train.element_spec}')

    def convert_dataset_to_spectrogram(self, dataset):
        return dataset.map(map_func = lambda audio, label: (self.get_spectrogram(audio), label),
                           num_parallel_calls=tf.data.AUTOTUNE
                        )
    
    def process_dataset(self):
        self.train_spectrograms = self.convert_dataset_to_spectrogram(self.train)
        self.validation_spectrograms = self.convert_dataset_to_spectrogram(self.validation)
        self.test_spectrograms = self.convert_dataset_to_spectrogram(self.test)

    def dataset_snapshot(self):
        print('Train Dataset................(Shapes and Length):', self.train_spectrograms.take(1), self.train_spectrograms.cardinality().numpy())
        print('Validation Dataset...........(Shapes and Length):', self.validation_spectrograms.take(1), self.validation_spectrograms.cardinality().numpy())
        print('Test Dataset.................(Shapes and Length):', self.test_spectrograms.take(1), self.test_spectrograms.cardinality().numpy())

    def get_train_dataset(self):
        return self.train_spectrograms.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    
    def get_validation_dataset(self):
        return self.validation_spectrograms.cache().prefetch(tf.data.AUTOTUNE)

    def get_test_dataset(self):
        return self.test_spectrograms.cache().prefetch(tf.data.AUTOTUNE)
    
    def get_shapes(self):
        for spectrogram, label in self.train_spectrograms.take(1):
            return (spectrogram[1].shape, label.shape)

    def load_dataset(self):
        self.train, self.validation = tf.keras.utils.audio_dataset_from_directory(f'{self.config.location.audios}',
                                                                             label_mode='categorical', 
                                                                             output_sequence_length=16000,
                                                                             batch_size=self.config.trainer.batch_size,
                                                                             validation_split=0.3,
                                                                             seed=42,
                                                                             subset='both',
                                                                        ) 
        
        self.labels_names = np.array(self.train.class_names)

        self.train = self.train.map(self.squeeze, tf.data.AUTOTUNE)
        self.validation= self.validation.map(self.squeeze, tf.data.AUTOTUNE)

        self.test = self.validation.shard(num_shards=2, index=0)
        self.validation = self.validation.shard(num_shards=2, index=1)