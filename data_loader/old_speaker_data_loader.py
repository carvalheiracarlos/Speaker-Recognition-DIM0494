import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from base.base_data_loader import BaseDataLoader

class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
        self.max_length = 0
    
    def get_max_output_sequence_length(self):
        max_duration = 0
        for root, dirs, files in os.walk(self.config.location.audios):
            for file in files:
                full_path = root + '/' + file
                print(f'Loading File.....:{full_path}')
                y, _ = librosa.load(full_path)
                aux = librosa.get_duration(y)
                if aux >= max_duration:
                    max_duration = aux
        self.max_length = max_duration
        return self.max_length
    
    def get_audio_from_directory(self):
        train_dataset = tf.keras.utils.\
            audio_dataset_from_directory(f'{self.config.location.audios}',
                                         label_mode='int', 
                                         output_sequence_length=6000,
                                         batch_size=self.config.trainer.batch_size,
                                         ) 
        train_dataset = train_dataset.map(librosa.feature.mfcc(sr=16000))
        return  train_dataset