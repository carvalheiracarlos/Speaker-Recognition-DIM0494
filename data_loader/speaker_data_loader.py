import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import librosa
import librosa.display
from sklearn import preprocessing

from base.base_data_loader import BaseDataLoader



class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
        self.labels = None 

    def generate_spectrograms(self):
        matplotlib.use('Agg')
        train_dataset = pd.read_csv(self.config.location.train, index_col=0)
        hop_length = 512 
        window_size = 1024
        for file_path in train_dataset['file_path']:
            file_name = file_path.split(os.path.sep)[-1].rsplit('.', 1)[0]

            y, sr = librosa.load(f'{self.config.location.root_folder}'+file_path)
            window = np.hanning(window_size)
            out  = librosa.core.spectrum.stft(y, n_fft = window_size, hop_length = hop_length, window=window)
            out = 2 * np.abs(out) / np.sum(window)

            fig = librosa.display.specshow(librosa.amplitude_to_db(out,ref=np.max))
            matplotlib.pyplot.xticks([])
            matplotlib.pyplot.yticks([])

            print(f'Saving File.........: {self.config.location.train_images}/' + file_name + '.jpg')
            matplotlib.pyplot.savefig(f'{self.config.location.train_images}/' + file_name + '.jpg', 
                                      transparent=True,
                                      bbox_inches='tight',
                                      pad_inches=0,
                                     )
            matplotlib.pyplot.cla()


    def label_encoder(self):
        train_dataset = pd.read_csv(self.config.location.train, index_col=0)
        self.labels = preprocessing.LabelEncoder().fit_transform(train_dataset['speaker'])
        return self.labels  

    def get_images_from_directory(self):
        self.labels = self.label_encoder()
        train_dataset = tf.keras.preprocessing.\
            image_dataset_from_directory(f'{self.config.location.images}',
                                         labels=self.labels.tolist(),
                                         label_mode='int', 
                                         batch_size=32, 
                                         image_size=(640, 480)
                                         ) 

        print(train_dataset.class_names)
        for batch_x, batch_y in train_dataset.take(1):
            print(batch_x.shape)
            print(batch_y.shape)

        return train_dataset

    
    def get_test_data(self):
        return self.X_test, self.y_test

