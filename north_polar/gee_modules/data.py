import ee
from typing import Dict, Any, Optional

from config import *



class FeatColl:
    def __init__(self, path: Optional[str] = None):
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


    def load(self) -> ee_featColl:
        """
        If path is valid, then load.
        """

        if self.validPath():
            try:
                fc = ee_featColl(self.path)
                return fc
            except Exception as e:
                raise ValueError(f"Failed to load FeatureCollection from path {self.path}") from e
        else:
            raise ValueError("Invalid path for FeatureCollection")
            
            
    def getInfo(self):
        """
        Get basic info of this collection
        + bounds
        + number of features
        """

        self.rectBounds = self.data.geometry().bounds()
        self.num_features = self.data.size()
            
    
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
        # Check if num_features is defined
        if not hasattr(self, 'num_features'):
            raise ValueError("FeatureCollection size is not defined. Call getInfo() first.")
        
        # Ensure num_features is an integer
        if not isinstance(self.num_features, int):
            if self.num_features is None:
                raise ValueError("FeatureCollection size is not defined")
            else:
                info = self.num_features.getInfo()
                if info is None:
                    raise ValueError("FeatureCollection size could not be retrieved (got None)")
                self.num_features = int(info)

        # Check if n is within the valid range
        if n < 0 or n >= self.num_features:
            raise IndexError("Index out of range")

        # Get the n-th feature
        return self.data.toList(1, n).get(0)