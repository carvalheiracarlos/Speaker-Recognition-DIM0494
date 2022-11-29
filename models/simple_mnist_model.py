from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten


class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
    
        self.model.add(Conv2D(filters=128, 
                         kernel_size=(5,5),
                         strides=(1,1),
                         activation='relu', 
                         input_shape=(1491, 257, 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(90, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )
