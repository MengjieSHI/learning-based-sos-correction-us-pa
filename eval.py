import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model
import matplotlib.pyplot as plt

from data_loader import *

# locate models
model_name = 'pureUSsimulations'
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, 'models/' + model_name)


trained_model = load_model(save_path+'./train.h5')
test_input = extract_signals('./us_simulated_rf_7.3_1511_5_layered.mat','non_filtered_rf_normalized')
test_output = trained_model.predict(test_input)

# visualisation
for i in range(0, 4):
    plt.subplot(2, 2, i+1)
    plt.imshow(np.rot90(tf.squeeze(test_output[i,:,:,:]), k=-1))
    clb=plt.colorbar(fraction=0.047)
    clb.ax.set_ylabel('SoS (m/s)')
plt.suptitle('Sound of Speed (SoS) Estimation')
plt.show()

