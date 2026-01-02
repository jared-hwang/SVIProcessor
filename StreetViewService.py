import requests
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

@dataclass
class StreetViewImage:
    """Data class representing a street view image"""
    image_data: bytes
    lat: Optional[float] = None
    lon: Optional[float] = None 
    heading: Optional[float] = None
    pitch: Optional[float] = None
    fov: Optional[float] = None
    timestamp: Optional[str] = None
    pano_id: Optional[str] = None
    dist_from_request: Optional[float] = None

class StreetViewService(ABC):
    """Abstract base class for street view services"""
    
    def __init__(self, name, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
    
    @abstractmethod
    def get_pano_with_id(self, id: str) -> Optional[StreetViewImage]:
        """Retrieve a street view image with a given ID"""
        pass
    
    @abstractmethod
    def get_pano_at_location(self, lat: float, lon: float, radius: float, **kwargs) -> Tuple[Optional[StreetViewImage], Dict[str, int]]:
        """
        Retrieve a street view image at/around a given latitude and longitude
        Returns: (StreetViewImage or None, stats dict with 'no_pano_locations' and 'download_failures')
        """
        pass 
    
    @abstractmethod
    def get_panos_around_location(self, lat: float, lon: float, radius: float, **kwargs) -> Tuple[List[StreetViewImage], Dict[str, int]]:
        """
        Retrieve street view images within the radius of a given latitude and longitude
        Returns: (List of StreetViewImages, stats dict with 'no_pano_locations' and 'download_failures')
        """
        pass
    
    @abstractmethod
    def get_panos_at_locations_batched(self, locations: List[Tuple[float, float]], radius: float, **kwargs) -> Tuple[List[StreetViewImage], Dict[str, int]]:
        """
        Downloads panos batchwise for efficiency, if possible
        Returns: (List of StreetViewImages, stats dict with 'no_pano_locations' and 'download_failures')
        """
        pass
    
    @abstractmethod
    def get_panos_around_locations_batched(self, locations: List[Tuple[float, float]], radius: float, **kwargs) -> Tuple[List[List[StreetViewImage]], Dict[str, int]]:
        """
        Downloads panos batchwise for efficiency, if possible
        Returns: (List of lists of StreetViewImages, stats dict with 'no_pano_locations' and 'download_failures')
        """
        pass