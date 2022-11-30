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
    
    def squeeze_datasets(self):
        self.train = self.train.map(self.squeeze, tf.data.AUTOTUNE)
        self.validation= self.validation.map(self.squeeze, tf.data.AUTOTUNE)

    def get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def inspect_spectrgoram(self):
        for audio, labels in self.train.take(1):  
            spectrogram = self.get_spectrogram(audio[1])
            print(f'Label Value................................: {self.labels_names[labels[1]]}')
            print(f'Label Shape(Batch Size, None)..............: {labels.shape}')
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
        for spectrogram, labels in self.train_spectrograms.take(1):
            print(f'Converted Dataset Spectrogram Shape........:{spectrogram[1].shape}')
            print(f'Converted Dataset Labels Shape(Batch Size, None)........:{labels.shape}')
            break

    def get_train_dataset(self):
        return self.train_spectrograms.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    
    def get_validation_dataset(self):
        return self.validation_spectrograms.cache().prefetch(tf.data.AUTOTUNE)

    def get_test_dataset(self):
        return self.test_spectrograms.cache().prefetch(tf.data.AUTOTUNE)
    
    def get_shapes(self):
        for spectrogram, label in self.train_spectrograms.take(1):
            return (spectrogram[1].shape, label.shape[0])

    def load_dataset(self):
        self.train, validation = tf.keras.utils.audio_dataset_from_directory(f'{self.config.location.audios}',
                                                                             label_mode='int', 
                                                                             output_sequence_length=16000,
                                                                             batch_size=self.config.trainer.batch_size,
                                                                             validation_split=0.2,
                                                                             seed=42,
                                                                             subset='both',
                                                                        ) 
        
        self.test = validation.shard(num_shards=2, index=0)
        self.validation = validation.shard(num_shards=2, index=1)
        self.labels_names = np.array(self.train.class_names)