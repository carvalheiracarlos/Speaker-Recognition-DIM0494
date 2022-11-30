from base.base_model import BaseModel
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Resizing, Input
from keras.metrics import categorical_accuracy


class SpeakerConv2D(BaseModel):
    def __init__(self, config, input_shape, n_labels):
        super(SpeakerConv2D, self).__init__(config)
        self.input_shape = input_shape
        self.n_labels = n_labels
        self.optimizer = Adam()
        self.build_model()

    def build_model(self):
        #print(self.input_shape)
        print(self.n_labels)

        self.model = Sequential()
        self.model.add(Input(shape=self.input_shape))
        self.model.add(Resizing(32, 32)) 
        self.model.add(Conv2D(32, 3, activation='relu')) 
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.n_labels))

        self.model.compile(
            loss=self.config.model.loss,
            optimizer=self.optimizer,
            metrics=[self.config.model.accuracy],
        )
