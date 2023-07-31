
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any, Union, Dict
from sklearn.base import BaseEstimator


class Stacking:

    """
    A class that implements the Stacking Algorithm for Survival Analysis.

    Methods:
    --------
    __init__():
        Initializes the Stacking instance.

    __check_input(X, y):
        Checks if the input data X and target y are in the correct format (pandas DataFrame/Series or numpy arrays).

    _stack_data(data, time_interval_column, censored=None, train=True):
        Method to stack data for survival analysis. The data are arranged according to the risk set at each unique time interval.

    fit(X, y, model, fit_params=None, censored=None):
        Fits a model on the stacked training data.

    predict(X, time_interval_list=None):
        Predicts the target for a given feature dataset.

    predict_proba(X, time_interval_list=None):
        Predicts the probabilities of the classes for a given feature dataset.

    build_survival_curve(new_observation, stacked_data=None, time_to_event=None, event_indicator=None):
        Builds and plots the survival curve for a new observation given the event indicator and the time to event.

    __plotign_function(list_of_intervals, survival_probabilities, standard_errors, figure_size=(10, 6)):
        Private method to plot the survival curve.

    __repr__():
        Returns a string representation of the Stacking instance.
    """

    def __init__(self):
        self.model = None
        self._stacked_train_data = None
        self._stacked_train_labels = None
        self.time_to_event_in_training = None
        self._stacked_test_data = None
        self._stacked_test_labels = None

    def __check_input(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Checks if the input data X and target y are in the correct format (pandas DataFrame/Series or numpy arrays).
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError('Input data should be a pandas DataFrame or a numpy array')
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError('Target data should be a pandas Series or a numpy array')

        return X, y

    def _stack_data(self, data: Union[pd.DataFrame, np.ndarray],
                    time_interval_column: Union[str, pd.Series],
                    censored: Optional[np.ndarray] = None, train=True) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Method to stack data for survival analysis. The data are arranged according
        to the risk set at each unique time interval.

        If 'censored' is None, it is assumed that all samples in the data are uncensored.

        Parameters:
            data (pd.DataFrame or np.ndarray): The dataset to stack.
            time_interval_column (str or pd.Series): The column of time intervals.
            censored (np.ndarray): Array of censoring indicators. If None, it is assumed
                                   that all samples are uncensored. Defaults to None.

        Returns:
            Tuple (np.ndarray, np.ndarray, np.ndarray): Tuple containing the stacked training data,
                                                        stacked training labels, and time to event in training data.
        """

        # Ensure input data is of correct type
        data, time_interval_column = self.__check_input(data, time_interval_column)

        data = np.array(data)

        # Assume all samples are uncensored if no censoring information is provided
        if censored is None:
            censored = np.ones(len(data))
        # Identify unique death times
        unique_death_times = set(time_interval_column)

        # Prepare a list for collecting DataFrame chunks
        stacked_data_list = []
        time_to_event = []

        for t in unique_death_times:
            # Get the risk set at this time
            if train:
                at_risk_indices = np.where(time_interval_column >= t)[0]
            else:
                at_risk_indices = np.where(time_interval_column == t)[0]

            # Get all the instances from data that are at risk at this time point
            risk_set = data[at_risk_indices]
            sub_set = [t] * len(risk_set)
            time_to_event.extend(sub_set)

            # Prepare the binary response: 1 for death and 0 for censored at this time
            binary_response = np.where((time_interval_column[at_risk_indices] == t) & (censored[at_risk_indices] == 1),
                                       1, 0)

            # Prepare a DataFrame for the risk set
            risk_set_df = pd.DataFrame(risk_set.copy())
            risk_set_df['y_binary'] = binary_response

            # For each uncensored death time, add a column indicating whether a it is in the risk set
            for t_prime in unique_death_times:
                risk_set_df[f'risk_set_{t_prime}'] = np.where(t == t_prime, 1, 0)

            # Append the DataFrame to the list
            stacked_data_list.append(risk_set_df)

        # Concatenate all DataFrame chunks at once
        stacked_df = pd.concat(stacked_data_list, axis=0)

        # Separate the features and the target
        self._stacked_train_data = np.array(stacked_df.drop('y_binary', axis=1).values)
        self._stacked_train_labels = np.array(stacked_df['y_binary'].values)
        self.time_to_event_in_training = np.array(time_to_event)

        return self._stacked_train_data, self._stacked_train_labels, self.time_to_event_in_training

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            model: BaseEstimator, fit_params: Optional[Dict[str, Any]] = None,
            censored: Optional[np.ndarray] = None) -> None:

        """
        Fits a model on the stacked training data. The input data is first stacked
        and then used for training the model.

        If 'censored' is None, it is assumed that all samples in the data are uncensored.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature dataset.
            y (pd.Series or np.ndarray): Target dataset.
            model (object): An instance of a classification model with fit and predict methods.
            fit_params (dict): Additional fitting parameters for the model. Defaults to None.
            censored (np.ndarray): Array of censoring indicators. If None, it is assumed
                                   that all samples are uncensored. Defaults to None.

        Raises:
            ValueError: If the model doesn't have fit or predict methods.
        """

        required_methods = ['fit', 'predict']
        if not all(hasattr(model, method) for method in required_methods):
            raise TypeError("Model must have 'fit' and 'predict' methods")

        X_stack, y_stack, time_to_event = self._stack_data(X, y, censored=censored)

        self.model = model

        fit_params = fit_params if fit_params is not None else {}

        self.model.fit(X_stack, y_stack, **fit_params)

    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                time_interval_list: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:

        """
        Predicts the target for a given feature dataset.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature dataset to predict the targets of.
            time_interval_list (pd.Series or np.ndarray): List of time intervals. Defaults to None.

        Returns:
            np.ndarray: Predicted targets for the input features.

        Raises:
            ValueError: If the model was not fitted before prediction.
        """

        if self.model is None:
            raise ValueError("You should fit the model first")

        if time_interval_list is not None:
            X, y, _ = self._stack_data(X, time_interval_list, train=False)

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], time_interval_list=None) -> np.ndarray:

        """
        Predicts the probabilities of the classes for a given feature dataset.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature dataset to predict the class probabilities of.

        Returns:
            np.ndarray: Predicted class probabilities for the input features.

        Raises:
            ValueError: If the model was not fitted before prediction.
            AttributeError: If the model doesn't have a predict_proba method.
        """

        if self.model is None:
            raise ValueError("You should fit the model first")

        if time_interval_list is not None:
            X, y, _ = self._stack_data(X, time_interval_list, train=False)

        try:
            pred = self.model.predict_proba(X)
        except AttributeError as e:
            return print(e)

        return pred

    def build_survival_curve(self, new_observation: np.ndarray,
                         stacked_data: Optional[np.ndarray] = None,
                         time_to_event: Optional[np.ndarray] = None,
                         event_indicator: Optional[np.ndarray] = None) -> None:
        """
        Builds and plots the survival curve for a new observation given the event
        indicator and the time to event.

        Parameters:
            new_observation (np.ndarray): New data observation to build survival curve for.
            stacked_data (np.ndarray): Stacked dataset. If None, uses stacked training data. Defaults to None.
            time_to_event (np.ndarray): Time to event data. If None, uses time to event in training data. Defaults to None.
            event_indicator (np.ndarray): Event indicator data. If None, uses stacked training labels. Defaults to None.

        Raises:
            ValueError: If the model was not fitted before building survival curve.
        """

        stacked_data = stacked_data or self._stacked_train_data
        time_to_event = time_to_event or self.time_to_event_in_training
        event_indicator = event_indicator or self._stacked_train_labels

        # Sort unique times to death
        unique_times = np.sort(np.unique(time_to_event))

        survival_probabilities = [1.0]  # starting with survival probability of 1 at t=0
        standard_errors = [0.0]  # starting with standard error of 0 at t=0
        list_of_intervals = [0]  # starting time at t=0

        for q, tq in enumerate(unique_times):
            # Get the indices for this stratum
            stratum_indices = np.where(time_to_event == tq)[0]

            # Compute the column mean and subvector mean for this stratum
            Mq = np.mean(stacked_data[stratum_indices], axis=0)
            alpha_hat_q = np.mean(event_indicator[stratum_indices])

            # Predict the conditional death probability
            period_subset_indicator = np.array([0] * len(unique_times))
            period_subset_indicator[q] = 1
            _new_observation = np.hstack((new_observation, period_subset_indicator))

            try:
                prediction = self.model.predict_proba(np.array([_new_observation - Mq]))[0][0]
            except AttributeError:
                prediction = self.model.predict(np.array([_new_observation - Mq]))[0]
                if isinstance(prediction, (np.ndarray)):
                    prediction = prediction[0]

            event_prob = alpha_hat_q + prediction
            # Compute survival probability and append to list
            survival_prob = 1 - event_prob
            survival_probabilities.append(survival_prob)

            # Compute standard error using Greenwood's formula
            nq = len(stratum_indices)
            yq = sum(event_indicator[stratum_indices])
            se = survival_prob * np.sqrt(yq / (nq * (nq - yq)))
            standard_errors.append(se)

            list_of_intervals.append(tq)  # add this interval to the list

        # Plot survival curve
        self.__plotign_function(list_of_intervals, survival_probabilities, standard_errors)


    def __plotign_function(self, list_of_intervals, survival_probabilities, standard_errors, figure_size=(10, 6)):
        plt.figure(figsize=figure_size)
        plt.step(list_of_intervals, survival_probabilities, where='post')

        # Plot confidence intervals
        survival_probabilities = np.array(survival_probabilities)
        standard_errors = np.array(standard_errors)
        plt.fill_between(list_of_intervals, survival_probabilities - 1.96 * standard_errors,
                         survival_probabilities + 1.96 * standard_errors, step='post', alpha=0.2)

        plt.xlabel('Time')
        plt.ylabel('Survival probability')
        plt.title('Survival Curve')
        plt.grid(True)
        plt.show()

    def __repr__(self):
        return f'Stacking API'

