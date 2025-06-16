import ee

from dataclasses import dataclass
from typing import Dict, Any

# alias for ee.FeatureCollection
FeatureCollection = ee.featurecollection.FeatureCollection

class FeatColl:
    def __init__(self, path:str = None):
        self.path = path
        self.data = self.load()
        self.getInfo()


    def validPath(self):
        """
        Make sure path is valid
        1. must be string
        2. must start with users/ or projects/
        """

        if not isinstance(self.path, str):
            raise ValueError("Path must be a string")
        
        if not (self.path.startswith("users/") or self.path.startswith("projects/")):
            raise ValueError("Path must start with 'users/' or 'projects/'")
        
        return True


    def load(self) -> FeatureCollection:
        """
        If path is valid, then load.
        """

        if self.validPath():
            try:
                fc = FeatureCollection(self.path)
                return fc
            except Exception as e:
                raise ValueError(f"Failed to load FeatureCollection from path {self.path}")
            
            
    def getInfo(self):
        """
        Get basic info of this collection
        + bounds
        + number of features
        """

        self.rectBounds = self.fc.geometry().bounds()
        self.num_features = self.fc.size()
            
    
    """
    -------------------------------------------------------------
    Note: Remote data transmission is required for methods below.

    Do not use these methods in a loop or repeatedly.
    -------------------------------------------------------------
    """
    def __get__(self, n:int):
        """
        Get n-th feature from this collection
        """

        if n < 0 or n >= self.num_features:
            raise IndexError("Index out of range")
        
        return self.fc.toList(1, n).get(0)