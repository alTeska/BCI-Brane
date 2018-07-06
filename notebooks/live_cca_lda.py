from filtering_cca import *
import training_cca_lda.py
import pickle

filename = 'LDA_model.sav'
classifier = pickle.load(open(filename, 'rb'))

r = cca_corrs(X_input, 15, 12, 8, sampling_rate)
y_est = classifier.predict(r)
print(y_est)