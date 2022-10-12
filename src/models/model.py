#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Model(ABC):
    """Parent class for easily wrapping models generated in projects in class, API and dashboard
    structure.
    
    Predefines necessary setters and getters and defines necessary interfaces for model import,
    prediction and visualization.
    """

    def __init__(self):
        """
        Initialize Model Object.
        """
        self._meta = {'name': '', 'description': '',
                      'data': {'example': {'min': 0, 'max': 1, 'fallback': .5}}}
        self._model = self._import_model()

    @abstractmethod
    def _predict(self, data):
        """
        Abstract method which is used to make predictions.

        Args:
            data (dict): Dict of named features to use in prediction.
        """
        pass

    @abstractmethod
    def _visualize(self, data):
        """
        Abstract method which is used to visualize targets/predictions.

        Args:
            data (dict): Dict of named features and targets/predictions to use in visualization.
        """
        pass

    def _import_model(self):
        """
        Create Placeholder for trained model pre-loading.

        Needs to be implemented per model.

        Returns:
            None.
        """
        return None

    def visualize_model(self, data=None):
        """
        Visualize model with given data. This method can optionally be overwritten if visualization
        of model is desired. Otherwise None is returned.

        Args:
            data (dict, optional): Dict of named features and targets/predictions to use in
                model-visualization. Defaults to None.
        """
        raise NotImplementedError

    def visualize_results(self, data=None):
        """
        Use given data to build prediction and visualize them.

        Args:
            data (dict, optional): Input data to pass to model. Fields not present are filled with
                fallback values. Defaults to None.

        Returns:
            Visualization of targets/predictions.
        """
        pred = self.prep_and_predict(data)
        data['pred'] = pred
        image = self._visualize(data)
        return image

    def get_meta_information(self, fields=None):
        """
        Get model meta information.

        Args:
            fields (list, optional): Meta information fields to get. Returns all if set to None.
                Defaults to None.

        Returns:
            dict: requested meta information.
        """
        if fields is None:
            return self._meta

        return {k: self._meta[k] for k in fields}

    def prep_and_predict(self, data=None):
        """
        Interface for prediction.
        
        Appends given data with fallback values and calls model-specific "predict"-method

        Args:
            data (dict, optional): Values to pass to model. Fields not present are filled with
                fallback values. Defaults to None.

        Returns:
            dict: Dictionary of named predictions.
        """
        if data is None:
            data = {}

        inp_data = {}
        for feature in self._meta['data']:
            if feature not in data:
                inp_data[feature] = self._meta['data'][feature]['fallback']
            else:
                inp_data[feature] = data[feature]

        prediction = self._predict(inp_data)
        return prediction
