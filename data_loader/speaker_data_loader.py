import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

from base.base_data_loader import BaseDataLoader

class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
    
    def label_encoder(self):
        train_dataset = pd.read_csv(self.config.location.train, index_col=0)
        #self.labels = preprocessing.LabelEncoder().fit_transform(train_dataset['speaker'])
        return train_dataset.speaker

    def get_audio_from_directory(self):
        labels = self.label_encoder()
        train_dataset = tf.keras.utils.\
            audio_dataset_from_directory(f'{self.config.location.audios}',
                                         label_mode='int', 
                                         output_sequence_length=60000,
                                         batch_size=8
                                         ) 
        return  train_dataset