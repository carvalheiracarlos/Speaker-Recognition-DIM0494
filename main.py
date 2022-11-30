from data_loader.speaker_data_loader import SpeakerDataLoader
from models.simple_mnist_model import SimpleMnistModel
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
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
    data_loader.squeeze_datasets()
    data_loader.inspect_spectrgoram()
    data_loader.process_dataset()
    data_loader.dataset_snapshot()

    '''
    print('Create the model.')
    model = SimpleMnistModel(config)
    print('Create the trainer')
    trainer = SimpleMnistModelTrainer(model.model, 
                                      train,
                                      config
                                    )
    print('Start training the model.')
    trainer.train()
    '''
if __name__ == '__main__':
    main()
