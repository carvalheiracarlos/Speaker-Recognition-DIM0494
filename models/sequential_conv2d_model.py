from base.base_model import BaseModel
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Conv2D, 
    BatchNormalization, MaxPooling2D, Dropout, 
    Flatten, Resizing, Input, BatchNormalization
)
from keras.metrics import categorical_accuracy


class SpeakerConv2D(BaseModel):
    def __init__(self, config, input_shape, n_labels):
        super(SpeakerConv2D, self).__init__(config)
        self.input_shape = input_shape
        self.n_labels = n_labels
        self.optimizer = Adam(learning_rate=self.config.model.learning_rate, 
                              beta_1=0.9, 
                              beta_2=0.999, 
                              epsilon=1e-08, 
                              decay=self.config.model.learning_rate/ self.config.trainer.num_epochs
                            )
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        self.model.add(Input(shape=self.input_shape))
        self.model.add(Resizing(64, 69)) 

        self.model.add(Conv2D(64, kernel_size=(8,8), strides=(2,2), activation='relu')) 
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, kernel_size=(4,4), strides=(2, 2), activation='relu')) 
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))


        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.n_labels[1], activation='softmax'))

        self.model.compile(
            loss=self.config.model.loss,
            optimizer=self.optimizer,
            metrics=[self.config.model.accuracy],
        )
