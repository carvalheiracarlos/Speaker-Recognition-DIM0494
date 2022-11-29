import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

from base.base_data_loader import BaseDataLoader

DATASET_SIZE = 1024 

class SpeakerDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SpeakerDataLoader, self).__init__(config)
        self.train = tf.data.Dataset.from_tensor_slices([])
        self.validation = tf.data.Dataset.from_tensor_slices([])
        self.test = tf.data.Dataset.from_tensor_slices([])
    
    def load_wav(self, file_path):
        file = tf.io.read_file(file_path)
        wav, sample_rate = tf.audio.decode_wav(file, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        return wav

    def preprocess(self, audio, label):
        for wav in audio:
            wav = wav[:48000]
            zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
            wav = tf.concat([zero_padding, wav],0)
            spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.expand_dims(spectrogram, axis=2)

            return spectrogram, label

    def wavs_as_datasets(self):
        aux_csv = pd.read_csv(self.config.location.train)
        spects_csv = pd.read_csv('./out.csv')
        files_paths = os.path.join(self.config.location.root_folder, 'train/*.wav')
        processed_files = self.preprocess()
        wav_files = tf.data.Dataset.from_tensor_slices(processed_files)
        labels = tf.data.Dataset.from_tensor_slices(aux_csv.speaker)
        dataset = tf.data.Dataset.zip((wav_files, labels))

        return dataset

    
    def create_pipeline(self):
        data = self.wavs_as_datasets()
        data = data.cache()
        data = data.shuffle(buffer_size=1000)
        data = data.batch(16)
        data = data.prefetch(8)
        
        train_size = int(0.7 * DATASET_SIZE)
        val_size = int(0.15 * DATASET_SIZE)
        test_size = int(0.15 * DATASET_SIZE)
        
        self.train = data.take(train_size)
        self.test = data.skip(train_size)
        self.validation = data.skip(val_size)
        self.test = data.take(test_size)

    def get_train(self):
        return self.train

    def get_validation(self):
        return self.validation

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

        return train_dataset


    def get_audio_from_directory(self):
        self.labels = self.label_encoder()
        train_dataset = tf.keras.utils.\
            audio_dataset_from_directory(f'{self.config.location.audios}',
                                         labels=self.labels.tolist(),
                                         label_mode='int', 
                                         batch_size=32,
                                         ) 
        return  train_dataset.batch(32)