import numpy as np 
predictions = np.squeeze(np.load('/home/andrew/Downloads/Ilk_fit_deep_predictions.npy'))
labels = np.squeeze(np.load('/home/andrew/Downloads/Ilk_fit_deep_test.npy'))

import pandas as pd 

output = pd.DataFrame({'predictions':predictions,'labels':labels})
output.to_csv('Ilk_fit_deep_eval.csv')