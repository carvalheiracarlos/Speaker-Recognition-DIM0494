from base.base_data_loader import BaseDataLoader
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal


class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)


    def generate_spectrograms(self):
        spectrograms = {
            'spectrogram': [],
            'labels': []
        }

        train_dataset = pd.read_csv(self.config.location.train, index_col=0)
        #sample = train_dataset['file_path'].apply(wavfile.read)
        spectrograms['labels'] = train_dataset['speaker']

        for file_path in train_dataset['file_path']:
            sample_rate, sample = wavfile.read(f'{self.config.location.root_folder}'+file_path)
            frequencies, times, spectrogram = signal.spectrogram(sample, sample_rate)
            spectrograms['spectrogram'].append(spectrogram)


        train_spectrograms = pd.DataFrame(spectrograms)
        print('DataSet Preview: \n', train_spectrograms)
        train_spectrograms.to_csv(self.config.location.train_processed)
    
    def get_train_data(self):
        X_train = pd.read_csv(self.config.data_location.train)
        layer = tf.keras.layers.CategoryEncoding(num_tokens=90, output_mode='one_hot')
        y_train = X_train['labels']
        print(X_train['Spectrograms'][1])
        return X_train, y_train,  

    def get_test_data(self):
        return self.X_test, self.y_test

