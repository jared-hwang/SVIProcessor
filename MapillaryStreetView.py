import requests
import geopy.distance
import mapillary
import mapillary.interface as mly
from typing import List, Tuple, Dict, Optional
from .StreetViewService import StreetViewService, StreetViewImage
import concurrent.futures

class MapillaryStreetView(StreetViewService):
    """Mapillary street-level imagery API implementation"""
    
    def __init__(self, name, api_key: str, verbose: bool = False):
        super().__init__(name, api_key)
        self._base_url = "https://graph.mapillary.com"
        self.verbose = verbose

    def get_pano_with_id(self, image_id: str) -> StreetViewImage:
        """Retrieve a street view image with a given ID"""
        # Mapillary Graph API endpoint for the image
        url = f'https://graph.mapillary.com/{image_id}'
        
        # Request image URL and metadata fields
        params = {
            'access_token': self.api_key,
            'fields': 'thumb_original_url,thumb_2048_url,id,computed_geometry,captured_at,computed_compass_angle'
        }

        if self.verbose:
            print(f"Fetching metadata for Image ID: {image_id}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Try to get the original URL, fall back to 2048px if unavailable
            image_url = data.get('thumb_original_url')
            if not image_url:
                if self.verbose:
                    print("Original URL not available, trying 2048px...")
                image_url = data.get('thumb_2048_url')
            
            if not image_url:
                if self.verbose:
                    print("Error: No download URL found in the API response.")
                return None

            # Download the actual image content
            if self.verbose:
                print(f"Downloading image from: {image_url}")
            img_response = requests.get(image_url, stream=True)
            img_response.raise_for_status()
            
            # Get the image data as bytes
            image_data = img_response.content
            
            # Extract metadata if available
            lat = None
            lon = None
            heading = None
            timestamp = None
            
            # Parse geometry coordinates if available
            if 'computed_geometry' in data:
                geometry = data['computed_geometry']
                if 'coordinates' in geometry:
                    lon = geometry['coordinates'][0]
                    lat = geometry['coordinates'][1]
            
            # Get compass angle (heading) if available
            if 'computed_compass_angle' in data:
                heading = data['computed_compass_angle']
            
            # Get timestamp if available
            if 'captured_at' in data:
                timestamp = data['captured_at']
            
            # Create and return StreetViewImage object
            return StreetViewImage(
                image_data=image_data,
                lat=lat,
                lon=lon,
                heading=heading,
                pitch=None,  # Mapillary doesn't provide pitch in standard API
                fov=None,    # Mapillary doesn't provide FOV in standard API
                timestamp=timestamp,
                pano_id=image_id
            )
            
        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"Error occurred: {e}")
            return None
    
    def _get_pano_metadata(self, lat: float, lon: float, radius: float, **kwargs):
        """
        Helper method to get pano metadata without downloading images.
        Returns the raw feature data from Mapillary API.
        """
        kwargs.pop('radius', None)
        features_geojson = mly.get_image_close_to(lat, lon, radius=radius, **kwargs)
        features_geojson = features_geojson.to_dict()
        
        # Filter to only panoramic images
        panos = [
            pano for pano in features_geojson['features']
            if pano['properties'].get('is_pano', False)
        ]
        
        return panos
        
    def get_pano_at_location(self, lat: float, lon: float, radius: float, 
                            silent: bool = False, 
                            existing_image_ids: set = None,
                            **kwargs) -> Tuple[Optional[StreetViewImage], Dict[str, int]]:
        """
        Retrieve a street view image at/around a given latitude and longitude        
        :param radius: radius around which to search
        :param silent: if True, suppress start/end print statements (used for batch operations)
        kwargs params
        :param **kwargs.min_date: minimum capture date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
        :param **kwargs.max_date: maximum capture date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
        :return: Tuple of (StreetViewImage, stats_dict)
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int}
        """
        if not silent:
            print(f"Starting get_pano_at_location...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        try:
            # Get metadata only
            panos = self._get_pano_metadata(lat, lon, radius, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"Error getting pano metadata: {e}")
            stats['no_pano_locations'] = 1
            if not silent:
                print(f"Finished get_pano_at_location. Panos downloaded: 0")
            return None, stats
        
        # Find the closest pano
        closest_pano = None
        closest_dist = float('inf')
        for pano in panos:
            coords = pano['geometry']['coordinates'][1], pano['geometry']['coordinates'][0]  # switch from lon, lat to lat, lon
            distance_from_request = geopy.distance.geodesic((lat, lon), coords).meters
            if distance_from_request < closest_dist:
                closest_dist = distance_from_request
                closest_pano = pano
        
        if not closest_pano:
            stats['no_pano_locations'] = 1
            if not silent:
                print(f"Finished get_pano_at_location. Panos downloaded: 0")
            return None, stats
        
        # Check if already downloaded
        pano_id = closest_pano['properties']['id']
        if pano_id in existing_image_ids:
            stats['skipped_existing'] = 1
            if not silent:
                print(f"Finished get_pano_at_location. Pano already downloaded, skipped.")
            
            # Return a skeleton SVI with just the ID and a flag
            skeleton_svi = StreetViewImage(
                image_data=None,  # No data since already downloaded
                lat=closest_pano['geometry']['coordinates'][1],
                lon=closest_pano['geometry']['coordinates'][0],
                heading=closest_pano['properties'].get('computed_compass_angle'),
                pitch=None,
                fov=None,
                timestamp=closest_pano['properties'].get('captured_at'),
                pano_id=pano_id
            )
            skeleton_svi.already_downloaded = True
            skeleton_svi.dist_from_request = closest_dist
            return skeleton_svi, stats
        
        # Download the closest pano
        svi = self.get_pano_with_id(pano_id)
        if svi:
            svi.dist_from_request = closest_dist
        else:
            stats['download_failures'] = 1
        
        panos_downloaded = 1 if svi else 0
        if not silent:
            print(f"Finished get_pano_at_location. Panos downloaded: {panos_downloaded}")
        return svi, stats

    def get_panos_around_location(self, lat: float, lon: float, radius: float, 
                              existing_image_ids: set = None,
                              **kwargs) -> Tuple[List[StreetViewImage], Dict[str, int]]:
        """
        Retrieve street view images within the radius of a given latitude and longitude
        
        :param lat: Latitude
        :param lon: Longitude
        :param radius: Search radius in meters
        :param existing_image_ids: Set of image IDs already downloaded (to skip)
        :param kwargs: Additional Mapillary API filters
        :return: Tuple of (list of StreetViewImage objects, stats_dict)
        """
        print(f"Starting get_panos_around_location...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        try:
            # Get metadata only
            panos = self._get_pano_metadata(lat, lon, radius, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"Error getting pano metadata: {e}")
            stats['no_pano_locations'] = 1
            print(f"Finished get_panos_around_location. Panos downloaded: 0")
            return [], stats
        
        if not panos:
            stats['no_pano_locations'] = 1
            print(f"Finished get_panos_around_location. Panos downloaded: 0")
            return [], stats
        
        svi_list = []
        svi_completed_ids = set()
        
        for pano in panos:
            pano_id = pano['properties']['id']
            
            # Skip if already downloaded
            if pano_id in existing_image_ids:
                stats['skipped_existing'] += 1
                if self.verbose:
                    print(f"Skipping already downloaded pano: {pano_id}")
                continue
            
            # Skip if we've already processed this ID in this batch
            if pano_id in svi_completed_ids:
                continue
            
            coords = pano['geometry']['coordinates'][1], pano['geometry']['coordinates'][0]
            distance_from_request = geopy.distance.geodesic((lat, lon), coords).meters
            
            # Download the image
            svi = self.get_pano_with_id(pano_id)
            if svi:
                svi.dist_from_request = distance_from_request
                svi_completed_ids.add(pano_id)
                svi_list.append(svi)
            else:
                stats['download_failures'] += 1

        print(f"Finished get_panos_around_location. Panos downloaded: {len(svi_list)}, "
            f"skipped existing: {stats['skipped_existing']}")
        return svi_list, stats

    def get_panos_at_locations_batched(self, locations: List[Tuple[float, float]], 
                                radius: float = 10, 
                                existing_image_ids: set = None,
                                **kwargs) -> Tuple[List[Optional[StreetViewImage]], Dict[str, int]]:
        """
        Downloads panos at multiple locations with parallel processing.
        Returns one pano per location (the closest one), maintaining position correspondence.
        
        :param locations: List of (lat, lon) tuples
        :param radius: Search radius for each location
        :param existing_image_ids: Set of image IDs already downloaded (to skip)
        :param kwargs: Additional Mapillary API filters
        :return: Tuple of (list of StreetViewImage objects or None, stats_dict)
                List maintains position correspondence - None for locations without panos
                stats_dict contains: {'no_pano_locations': int, 'download_failures': int}
        """
        print(f"Starting get_panos_at_locations_batched...")
        if self.verbose:
            print(f"Fetching panos at {len(locations)} locations...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        # Initialize results list with None to maintain position correspondence
        results = [None] * len(locations)
        
        # Use ThreadPoolExecutor for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all location queries with silent=True to suppress individual prints
            future_to_idx = {
                executor.submit(self.get_pano_at_location, lat, lon, radius, 
                            silent=True, existing_image_ids=existing_image_ids, **kwargs): idx
                for idx, (lat, lon) in enumerate(locations)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                location = locations[idx]
                try:
                    svi, location_stats = future.result()
                    # Place result at correct index (None if no pano found)
                    results[idx] = svi
                    
                    # Aggregate stats
                    stats['no_pano_locations'] += location_stats['no_pano_locations']
                    stats['download_failures'] += location_stats['download_failures']
                    stats['skipped_existing'] += location_stats.get('skipped_existing', 0)
                    
                    if not svi and self.verbose:
                        print(f"No pano found at location {location}")
                except Exception as e:
                    results[idx] = None  # Explicitly set to None on error
                    stats['no_pano_locations'] += 1
                    if self.verbose:
                        print(f"Error fetching pano at {location}: {e}")
        
        # Count successful downloads (non-None entries)
        successful_downloads = sum(1 for r in results if r is not None)
        
        print(f"Finished get_panos_at_locations_batched. Panos downloaded: {successful_downloads}")
        print(f"Stats: {stats['no_pano_locations']} locations without panos, "
            f"{stats['skipped_existing']} skipped (already downloaded), "
            f"{stats['download_failures']} download failures")
        return results, stats
    
    def get_panos_around_locations_batched(self, locations: List[Tuple[float, float]], 
                                       radius: float = 10,
                                       existing_image_ids: set = None,
                                       **kwargs) -> Tuple[List[List[StreetViewImage]], Dict[str, int]]:
        """
        Retrieve panoramas around multiple locations with automatic deduplication, using parallel downloading. 
        Each unique panorama is downloaded only once, then distributed to all relevant location results.
        
        :param locations: List of (lat, lon) coordinate tuples to search around
        :param radius: Search radius in meters for each location (default: 10)
        :param **kwargs: Additional Mapillary API filters (min_date, max_date, etc.)
        
        :return: Tuple of (list of lists, stats_dict)
                 List of lists where each inner list contains StreetViewImage objects
                 found around the corresponding input location. Order matches input 
                 locations. Empty list returned for locations with no panoramas.
                 stats_dict contains: {'no_pano_locations': int, 'download_failures': int}
        """
        print(f"Starting get_panos_around_locations_batched...")
        if self.verbose:
            print(f"Fetching panos around {len(locations)} locations...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        # First, collect all unique pano IDs across all locations
        all_pano_data = {}  # pano_id -> (pano_feature, locations_indices)
        location_panos = [[] for _ in locations]
        locations_with_no_panos = set()
        
        # Step 1: Gather all pano metadata with deduplication
        if self.verbose:
            print("Step 1: Gathering pano metadata...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(self._get_pano_metadata, lat, lon, radius, **kwargs): idx
                for idx, (lat, lon) in enumerate(locations)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    panos_data = future.result()
                    if not panos_data:
                        locations_with_no_panos.add(idx)
                    else:
                        for pano_feature in panos_data:
                            pano_id = pano_feature['properties']['id']
                            if pano_id not in all_pano_data:
                                all_pano_data[pano_id] = (pano_feature, [])
                            all_pano_data[pano_id][1].append(idx)
                except Exception as e:
                    locations_with_no_panos.add(idx)
                    if self.verbose:
                        print(f"Error fetching metadata for location {idx}: {e}")
        
        stats['no_pano_locations'] = len(locations_with_no_panos)
        
        # Filter out already downloaded panos
        panos_to_download = {
            pid: data for pid, data in all_pano_data.items() 
            if pid not in existing_image_ids
        }
        stats['skipped_existing'] = len(all_pano_data) - len(panos_to_download)
        
        if self.verbose:
            print(f"Found {len(all_pano_data)} unique panos across all locations")
            print(f"Skipping {stats['skipped_existing']} already downloaded panos")
            print(f"Will download {len(panos_to_download)} new panos")
        
        # Step 2: Download only new panos in parallel
        if self.verbose:
            print("Step 2: Downloading new panos...")
        pano_cache = {}
        
        if panos_to_download:
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_pano_id = {
                    executor.submit(self.get_pano_with_id, pano_id): pano_id
                    for pano_id in panos_to_download.keys()
                }
                
                for future in concurrent.futures.as_completed(future_to_pano_id):
                    pano_id = future_to_pano_id[future]
                    try:
                        svi = future.result()
                        if svi:
                            pano_cache[pano_id] = svi
                        else:
                            stats['download_failures'] += 1
                    except Exception as e:
                        stats['download_failures'] += 1
                        if self.verbose:
                            print(f"Error downloading pano {pano_id}: {e}")
        
        # Step 3: Distribute panos to their respective locations (only newly downloaded ones)
        if self.verbose:
            print("Step 3: Organizing results by location...")
        for pano_id, (pano_feature, location_indices) in panos_to_download.items():
            if pano_id in pano_cache:
                svi = pano_cache[pano_id]
                
                for idx in location_indices:
                    # Calculate distance for this specific location
                    lat, lon = locations[idx]
                    coords = (
                        pano_feature['geometry']['coordinates'][1],
                        pano_feature['geometry']['coordinates'][0]
                    )
                    distance_from_request = geopy.distance.geodesic((lat, lon), coords).meters
                    
                    # Create a copy with the specific distance
                    svi_copy = StreetViewImage(
                        image_data=svi.image_data,
                        lat=svi.lat,
                        lon=svi.lon,
                        heading=svi.heading,
                        pitch=svi.pitch,
                        fov=svi.fov,
                        timestamp=svi.timestamp,
                        pano_id=svi.pano_id
                    )
                    svi_copy.dist_from_request = distance_from_request
                    location_panos[idx].append(svi_copy)
        
        total_panos_downloaded = len(pano_cache)
        print(f"Finished get_panos_around_locations_batched. Panos downloaded: {total_panos_downloaded}")
        print(f"Stats: {stats['no_pano_locations']} locations without panos, "
            f"{stats['skipped_existing']} skipped (already downloaded), "
            f"{stats['download_failures']} download failures")
        return location_panos, stats

# Example usage:
# import os
# from dotenv import load_dotenv
# load_dotenv()
# MAPILLARY_TOKEN = os.getenv('MAPILLARY_TOKEN')
# mapillary.utils.auth.set_token(MAPILLARY_TOKEN)

# # Create with verbose=False for minimal output
# test = MapillaryStreetView(MAPILLARY_TOKEN, verbose=False)
# Or with verbose=True for detailed output
# test = MapillaryStreetView(MAPILLARY_TOKEN, verbose=True)

# # Single location
# svi = test.get_pano_at_location(47.6569533, -122.3132275, 10)
# with open('test.png', 'wb') as f:
#     f.write(svi.image_data)

# # Multiple panos around a location
# svi_list = test.get_panos_around_location(47.6569533, -122.3132275, 10)
# for i, svi in enumerate(svi_list):
#     with open(f'test_panos/test_{i}.png', 'wb') as f:
#         f.write(svi.image_data)

# Batch processing for multiple locations
# locations = [
#     (47.6569533, -122.3132275),
#     (47.6580113, -122.3147812),
#     (47.655801424438664, -122.30706602847431),
# ]
# batch_results = test.get_panos_around_locations_batched(locations, radius=10)

# for batch_num, batch in enumerate(batch_results):
#     for pano_num, pano in enumerate(batch):
#         with open(f'test_panos/test_{batch_num}_{pano_num}.png', 'wb') as f:
#             f.write(pano.image_data)