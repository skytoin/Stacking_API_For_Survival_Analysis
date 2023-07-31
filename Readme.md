# Stacking API

# Introduction 
This Python module contains a single class, Stacking, designed to implement the Stacking Algorithm for Survival Analysis. 
The goal of this class is to allow users to fit and apply conventional classification models to survival analysis data. 

Class: Stacking
This class contains several methods including initialization, data validation, data stacking for survival analysis, 
model fitting, predictions, and building survival curves.

Initialization (init)
The __init__ method initializes an instance of the Stacking class. This method doesn't take any arguments.

Usage:
codestacking_instance = Stacking()

Data Validation (__check_input)
This private method checks if the input data X and target y are in the correct format (pandas DataFrame/Series or numpy arrays).

Data Stacking (_stack_data)
The _stack_data method arranges data according to the risk set at each unique time interval.

Usage:
stacking_instance._stack_data(data, time_interval_column, censored, train)


Model Fitting (fit)
The fit method is used to train the model on the stacked data. If you have censored data then you have to build numpy array that contains 
1 for uncensored samples and 0 for censored samples. If you don't provide such array then the assumption is all your data is uncensored.

Usage
stacking_instance.fit(X, y, model, fit_params, censored)


Predictions (predict)
The predict method is used to predict the target for a given feature dataset.

Usage:
predictions = stacking_instance.predict(X, time_interval_list)


Predictions (predict_proba)
The predict_proba method is used to predict the probabilities of the classes for a given feature dataset. You have to make sure if 
the selected model has .predict_proba method.

Usage:
predictions = stacking_instance.predict_proba(X, time_interval_list)

Survival Curve Building (build_survival_curve)
The build_survival_curve method is used to build and plot the survival curve for a new observation given the event indicator and the time to event.

Usage:

stacking_instance.build_survival_curve(new_observation, stacked_data, time_to_event, 	event_indicator)


Requirements
This module requires Python 3.5+, NumPy, pandas, matplotlib, and scikit-learn.


Installation
The module does not require any special installation steps. Simply import the module in your Python environment.

Examples
The following is an example of how to use this class.

# first example
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Initialize Stacking instance
stacking = Stacking()

# Generate some random data
X = np.random.rand(300, 3)
y = np.random.randint(0, 10, size=300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
rf_model = RandomForestClassifier()
stacking.fit(X_train, y_train, rf_model)

# Make predictions
predictions = stacking.predict(X_test, y_test)

# Predict probabilities
prob_predictions = stacking.predict_proba(X_test, y_test)

# Build survival curve
new_observation = X_test[0]
stacking.build_survival_curve(new_observation)




# second example with neural network

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

# Initialize Stacking instance
stacking = Stacking()

# Generate some random data
X = np.random.rand(300, 3)
y = np.random.randint(0, 10, size=300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
model = Sequential([tf.keras.layers.Dense(32,  activation='relu'),
                               tf.keras.layers.Dense(16, activation='relu'),
                               tf.keras.layers.Dense(1, activation='sigmoid')])

# you have to compile the model before inputing it in Stacking instance
model.compile(loss=tf.keras.losses.binary_crossentropy,
                         optimizer=Adam(),
                         metrics=['accuracy'])

# training parameters
fit_params = {'epochs': 10, 'verbose': 0}
stacking.fit(X_train, y_train, model)


# Make predictions
predictions = stacking.predict(X_test, y_test)

# Build survival curve
new_observation = X_test[0]
stacking.build_survival_curve(new_observation)

