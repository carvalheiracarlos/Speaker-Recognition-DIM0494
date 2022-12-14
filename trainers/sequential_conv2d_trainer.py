from base.base_trainer import BaseTrain
import os
from keras.callbacks import ( 
    ModelCheckpoint, 
    TensorBoard, 
    CSVLogger 
)


class SpeakerConv2DModelTrainer(BaseTrain):
    def __init__(self, model, train_dataset, validation_dataset, config):
        super(SpeakerConv2DModelTrainer, self).__init__(model, train_dataset, validation_dataset, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.config.callbacks.model_weights,
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            CSVLogger(filename=self.config.callbacks.csv_logger, separator=",", append=False)
        )

    def train(self):
        self.model.summary()
        history = self.model.fit(self.train_dataset,
                                 validation_data=self.validation_dataset,
                                 epochs=self.config.trainer.num_epochs,
                                 verbose=self.config.trainer.verbose_training,
                                 callbacks=self.callbacks,
                            )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['categorical_accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_categorical_accuracy'])
