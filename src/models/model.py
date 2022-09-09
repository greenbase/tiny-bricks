#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:39:20 2022.

@author: brede
"""

class Model:
    """Parent class for easily wrapping models generated in projects in class, API and Dashboard structure.
    
    Predefines necessary setters and getters and defines necessary interfaces for model import, prediction and visualization.
    """
    
    def __init__(self, model, meta={'name': '', 'description': '', 'data': {'example': {'min': 0, 'max': 1, 'fallback': .5}}}):
        """
        Initialize Model Object.

        Args:
            meta (dict, optional): Model information. Includes model name, description and viable range of data. Defaults to {'name': '', 'description': '', 'data': {}}.

        Returns:
            None.

        """
        self.meta = meta
        self.model = self.import_model()
        
    def import_model(self):
        """
        Create Placeholder for trained model pre-loading.
        
        Needs to be implemented per model.

        Returns:
            None.

        """
        return None
    
    def predict(self, data):
        """
        Create Placeholder for prediction.

        Args:
            data (dict): Dict of named features to use in prediction.

        Returns:
            None.

        """
        return None
    
    def visualize(self, data):
        """
        Create Placeholder for vizualisation.

        Args:
            data (dict): Dict of named features and targets/predictions to use in visualization.

        Returns:
            None.

        """
        return None
    
    def visualize_model(self, data):
        """
        Create Placeholder for model-vizualisation.

        Args:
            data (dict): Dict of named features and targets/predictions to use in model-visualization.

        Returns:
            None.

        """
        return None
    
    def visualize_results(self, data=None):
        """
        Use data to build prediction, combine the results with the features and returns vis.

        Args:
            data (dict): Values to pass to model. Fields not present are filled with fallback values. Defaults to None.

        Returns:
            Values passed to model and it's prediction in the "pred" field.

        """
        pred = self.prep_and_predict(data)
        data['pred'] = pred
        return data
        
    def get_meta_information(self, fields = None):
        """
        Get Model-information.

        Args:
            fields (list, optional): Meta-Information fields to get. Returns all if set to None. Defaults to None.

        Returns:
            dict: requested Meta-Information.

        """
        if fields is not None:
            fields = list(self.meta.keys())
     
        return {k: self.meta[k] for k in fields}
    
    def prep_and_predict(self, data = None):
        """
        Interface for prediction.
        
        Appends given data with fallback values and calls model-specific "predict"-method

        Args:
            data (dict, optional): Values to pass to model. Fields not present are filled with fallback values. Defaults to None.

        Returns:
            None.

        """
        data = {k:data[k] if k in data else self.meta['data'][k]['fallback'] for k in self.meta['data']}
        prediction = self.predict(data)
        return prediction