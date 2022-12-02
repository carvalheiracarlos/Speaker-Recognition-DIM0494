import pandas as pd
import numpy as np



def save_predictions_to_csv(config, predictions, true_labels):
    pd.DataFrame(data={'predictions':np.argmax(predictions, axis=-1), 
                       'true_labels': np.argmax(true_labels, axis=-1)})\
                       .to_csv(config.location.predictions)