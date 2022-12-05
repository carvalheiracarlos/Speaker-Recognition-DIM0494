import pandas as pd
import numpy as np



def save_predictions_to_csv(config, predictions, true_labels):
    pd.DataFrame(data={'predictions':np.argmax(predictions, axis=-1), 
                       'true_labels': np.argmax(true_labels, axis=-1)})\
                       .to_csv(config.location.predictions)
                
def save_predictions_to_kaggle(config, predictions):
    kaggle_csv = pd.read_csv(config.location.kaggle, index_col=0)
    kaggle_csv.speaker = np.argmax(predictions, axis=-1)
    kaggle_csv.to_csv(config.location.kaggle)