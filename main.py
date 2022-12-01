from data_loader.speaker_data_loader import SpeakerDataLoader
from models.sequential_conv2d_model import SpeakerConv2D
from trainers.sequential_conv2d_trainer import SpeakerConv2DModelTrainer 
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import tensorflow as tf
import os

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the data generator.')
    data_loader = SpeakerDataLoader(config)
    data_loader.load_dataset()
    data_loader.inspect_dataset()
    data_loader.inspect_spectrgoram()
    data_loader.process_dataset()
    data_loader.dataset_snapshot()
    data_shapes = data_loader.get_shapes()

    if args.train:
        print('Create the model.')
        speaker_conv2d = SpeakerConv2D(config, data_shapes[0], data_shapes[1])
        print('Create the trainer')
        speaker_conv2d_trainer = SpeakerConv2DModelTrainer(speaker_conv2d.model, data_loader.get_train_dataset(), data_loader.get_validation_dataset(), config)
        print('Start training the model.')
        speaker_conv2d_trainer.train()


if __name__ == '__main__':
    main()
