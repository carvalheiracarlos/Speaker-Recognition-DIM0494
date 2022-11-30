import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from base.base_data_loader import BaseDataLoader

class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
        self.train = tf.data.Dataset.from_tensor_slices([]) 
        self.validation = tf.data.Dataset.from_tensor_slices([])

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
            print(audio.shape)
            print(labels.shape)

    def inspect_dataset(self):
        print('...........Audio Dataset..........')
        print(f'Classes Names.....: \n{np.array(self.train.class_names)}')
        print(f'Element Inpect....: \n{self.train.element_spec}')

    def load_dataset(self):
        self.train, self.validation = tf.keras.utils.audio_dataset_from_directory(f'{self.config.location.audios}',
                                                                                 label_mode='int', 
                                                                                 output_sequence_length=16000,
                                                                                 batch_size=self.config.trainer.batch_size,
                                                                                 validation_split=0.2,
                                                                                 seed=42,
                                                                                 subset='both',
                                                                                 ) 