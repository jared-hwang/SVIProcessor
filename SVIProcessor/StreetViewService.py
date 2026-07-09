import requests
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

@dataclass
class StreetViewImage:
    """Data class representing a street view image"""
    image_data: Optional[bytes]  # Made optional to support skeleton SVIs
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
    def get_pano_with_id(self, image_id: str) -> Optional[StreetViewImage]:
        """Retrieve a street view image with a given ID"""
        pass
    
    @abstractmethod
    def get_pano_at_location(self, lat: float, lon: float, radius: float, 
                            silent: bool = False, 
                            existing_image_ids: set = None,
                            **kwargs) -> Tuple[Optional[StreetViewImage], Dict[str, int]]:
        """
        Retrieve a street view image at/around a given latitude and longitude
        
        :param lat: Latitude coordinate
        :param lon: Longitude coordinate
        :param radius: Search radius in meters
        :param silent: If True, suppress start/end print statements (used for batch operations)
        :param existing_image_ids: Set of image IDs already downloaded to skip
        :param **kwargs: Additional service-specific parameters
        :return: Tuple of (StreetViewImage or None, stats_dict)
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int, 'skipped_existing': int}
        """
        pass 
    
    @abstractmethod
    def get_panos_around_location(self, lat: float, lon: float, radius: float, 
                                 existing_image_ids: set = None,
                                 **kwargs) -> Tuple[List[StreetViewImage], Dict[str, int]]:
        """
        Retrieve street view images within the radius of a given latitude and longitude
        
        :param lat: Latitude coordinate
        :param lon: Longitude coordinate
        :param radius: Search radius in meters
        :param existing_image_ids: Set of image IDs already downloaded to skip
        :param **kwargs: Additional service-specific parameters
        :return: Tuple of (List of StreetViewImages, stats_dict)
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int, 'skipped_existing': int}
        """
        pass
    
    @abstractmethod
    def get_panos_at_locations_batched(self, locations: List[Tuple[float, float]], 
                                      radius: float,
                                      existing_image_ids: set = None,
                                      **kwargs) -> Tuple[List[Optional[StreetViewImage]], Dict[str, int]]:
        """
        Downloads panos batchwise for efficiency, if possible
        
        :param locations: List of (lat, lon) coordinate tuples
        :param radius: Search radius in meters for each location
        :param existing_image_ids: Set of image IDs already downloaded to skip
        :param **kwargs: Additional service-specific parameters
        :return: Tuple of (List of StreetViewImages or None, stats_dict)
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int, 'skipped_existing': int}
        """
        pass
    
    @abstractmethod
    def get_panos_around_locations_batched(self, locations: List[Tuple[float, float]], 
                                          radius: float = 10,
                                          existing_image_ids: set = None,
                                          **kwargs) -> Tuple[List[List[StreetViewImage]], Dict[str, int]]:
        """
        Retrieve panoramas around multiple locations with automatic deduplication, using parallel downloading.
        Each unique panorama is downloaded only once, then distributed to all relevant location results.
        
        :param locations: List of (lat, lon) coordinate tuples to search around
        :param radius: Search radius in meters for each location (default: 10)
        :param existing_image_ids: Set of image IDs already downloaded to skip
        :param **kwargs: Additional service-specific parameters
        :return: Tuple of (List of lists of StreetViewImages, stats_dict)
                 List of lists where each inner list contains StreetViewImage objects
                 found around the corresponding input location. Order matches input
                 locations. Empty list returned for locations with no panoramas.
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int, 'skipped_existing': int}
        """
        pass