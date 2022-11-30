from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.metrics import categorical_accuracy


class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
    
        self.model.add(Conv2D(filters=32, 
                         kernel_size=(3,3),
                         strides=(1,1),
                         activation='relu', 
                         input_shape=(self.config.trainer.batch_size, 6000, 1)))
        self.model.add(Conv2D(filters=64, 
                         kernel_size=(2,2),
                         strides=(1,1),
                         activation='relu', ))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(12, activation='softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['accuracy'],
        )
