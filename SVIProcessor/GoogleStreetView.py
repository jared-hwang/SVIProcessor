import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import io
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import math
from streetlevel import streetview
import threading
from datetime import datetime
from .StreetViewService import StreetViewService, StreetViewImage


class GoogleStreetView(StreetViewService):
    """Google Street View API implementation"""
    
    def __init__(self, name="GoogleStreetView", api_key: Optional[str] = None, verbose: bool = False):
        super().__init__(name, api_key)
        self.verbose = verbose
        self._file_lock = threading.Lock()
        self._tile_session = self._make_tile_session()

    @staticmethod
    def _make_tile_session() -> requests.Session:
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session
    
    def get_pano_with_id(self, image_id: str) -> Optional[StreetViewImage]:
        """
        Retrieve a street view image with a given ID (panorama ID)
        Downloads and assembles the panorama from tiles
        """
        if self.verbose:
            print(f"Fetching panorama with ID: {image_id}...")
        
        # Use the fetch_panorama logic from download.py
        image_data = self._fetch_panorama(image_id)
        
        if image_data is None:
            if self.verbose:
                print(f"Failed to fetch panorama with ID: {image_id}")
            return None
        
        # Convert cv2 BGR image to bytes (as JPEG or PNG)
        success, buffer = cv2.imencode('.jpg', image_data)
        if not success:
            return None
        
        image_bytes = buffer.tobytes()
        
        # Try to get metadata using streetlevel library if available
        try:
            pano = streetview.find_panorama_by_id(image_id)
            if pano:
                return StreetViewImage(
                    image_data=image_bytes,
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=pano.heading,
                    pitch=pano.pitch,
                    fov=None,  # Google doesn't provide FOV in this API
                    timestamp=str(pano.date) if pano.date else None,
                    pano_id=image_id
                )
        except:
            pass
        
        # Return with just the image data if metadata fetch fails
        return StreetViewImage(
            image_data=image_bytes,
            pano_id=image_id
        )
    
    def get_pano_at_location(self, lat: float, lon: float, radius: float,
                            silent: bool = False,
                            existing_image_ids: set = None,
                            **kwargs) -> Tuple[Optional[StreetViewImage], Dict[str, int]]:
        """
        Retrieve a street view image at/around a given latitude and longitude
        """
        if not silent and self.verbose:
            print(f"Starting get_pano_at_location for ({lat}, {lon})...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        try:
            # Use streetlevel library to find panorama
            pano = streetview.find_panorama(lat=lat, lon=lon, radius=radius)
            
            if pano is None:
                stats['no_pano_locations'] = 1
                if not silent and self.verbose:
                    print(f"No panorama found at location ({lat}, {lon})")
                return None, stats
            
            # Check if already downloaded
            if pano.id in existing_image_ids:
                stats['skipped_existing'] = 1
                if not silent and self.verbose:
                    print(f"Panorama {pano.id} already downloaded, skipping.")
                
                # Return skeleton SVI
                skeleton_svi = StreetViewImage(
                    image_data=None,
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=pano.heading,
                    pitch=pano.pitch,
                    fov=None,
                    timestamp=str(pano.date) if pano.date else None,
                    pano_id=pano.id
                )
                skeleton_svi.already_downloaded = True
                skeleton_svi.dist_from_request = self._haversine(lat, lon, pano.lat, pano.lon)
                return skeleton_svi, stats
            
            # Download the panorama
            image_data = self._fetch_panorama(pano.id)
            
            if image_data is None:
                stats['download_failures'] = 1
                if not silent and self.verbose:
                    print(f"Failed to download panorama {pano.id}")
                # return skeleton so caller can distinguish from "no pano found"
                skeleton = StreetViewImage(
                    image_data=None,
                    lat=pano.lat, lon=pano.lon,
                    heading=pano.heading, pitch=pano.pitch, fov=None,
                    timestamp=str(pano.date) if pano.date else None,
                    pano_id=pano.id
                )
                skeleton.dist_from_request = self._haversine(lat, lon, pano.lat, pano.lon)
                return skeleton, stats

            # Convert to bytes
            success, buffer = cv2.imencode('.jpg', image_data)
            if not success:
                stats['download_failures'] = 1
                skeleton = StreetViewImage(
                    image_data=None,
                    lat=pano.lat, lon=pano.lon,
                    heading=pano.heading, pitch=pano.pitch, fov=None,
                    timestamp=str(pano.date) if pano.date else None,
                    pano_id=pano.id
                )
                skeleton.dist_from_request = self._haversine(lat, lon, pano.lat, pano.lon)
                return skeleton, stats
            
            image_bytes = buffer.tobytes()
            
            # Create StreetViewImage
            svi = StreetViewImage(
                image_data=image_bytes,
                lat=pano.lat,
                lon=pano.lon,
                heading=pano.heading,
                pitch=pano.pitch,
                fov=None,
                timestamp=str(pano.date) if pano.date else None,
                pano_id=pano.id
            )
            svi.dist_from_request = self._haversine(lat, lon, pano.lat, pano.lon)
            
            if not silent and self.verbose:
                print(f"Successfully downloaded panorama {pano.id}")
            
            return svi, stats
            
        except Exception as e:
            if self.verbose:
                print(f"Error processing location ({lat}, {lon}): {e}")
            stats['download_failures'] = 1
            return None, stats
    
    def get_panos_around_location(self, lat: float, lon: float, radius: float,
                                 existing_image_ids: set = None,
                                 **kwargs) -> Tuple[List[StreetViewImage], Dict[str, int]]:
        """
        Retrieve street view images within the radius of a given latitude and longitude
        Note: Google Street View API doesn't have a direct way to get multiple panoramas
        around a location, so this implementation only returns the closest one.
        """
        if self.verbose:
            print(f"Getting panoramas around ({lat}, {lon}) with radius {radius}m...")
        
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        
        # For Google Street View, we can only get the closest panorama
        # To get multiple, you'd need to implement a grid search or use historical data
        svi, single_stats = self.get_pano_at_location(
            lat, lon, radius, silent=True, existing_image_ids=existing_image_ids, **kwargs
        )
        
        stats.update(single_stats)
        
        if svi is not None:
            return [svi], stats
        else:
            return [], stats
    
    def get_panos_at_locations_batched(self, locations: List[Tuple[float, float]],
                                      radius: float = 10,
                                      existing_image_ids: set = None,
                                      **kwargs) -> Tuple[List[Optional[StreetViewImage]], Dict[str, int]]:
        """
        Downloads panoramas batchwise for efficiency using concurrent processing
        """
        print(f"Starting batch download for {len(locations)} locations...")
        
        existing_image_ids = existing_image_ids or set()
        stats = {'no_pano_locations': 0, 'download_failures': 0, 'skipped_existing': 0}
        results = [None] * len(locations)
        
        max_workers = kwargs.get('max_workers', 12)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.get_pano_at_location,
                    lat, lon, radius,
                    silent=True,
                    existing_image_ids=existing_image_ids,
                    **kwargs
                ): idx
                for idx, (lat, lon) in enumerate(locations)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    svi, location_stats = future.result()
                    results[idx] = svi
                    
                    # Update stats
                    for key in stats:
                        stats[key] += location_stats.get(key, 0)
                        
                except Exception as e:
                    lat, lon = locations[idx]
                    if self.verbose:
                        print(f"Error fetching panorama at location {idx} ({lat}, {lon}): {e}")
                    stats['download_failures'] += 1
        
        successful_downloads = sum(1 for r in results if r is not None and r.image_data is not None)
        print(f"Finished batch download. Panoramas downloaded: {successful_downloads}")
        print(f"Stats: {stats['no_pano_locations']} locations without panoramas, "
              f"{stats['skipped_existing']} skipped (already downloaded), "
              f"{stats['download_failures']} download failures")
        
        return results, stats
    
    def get_panos_around_locations_batched(self, locations: List[Tuple[float, float]],
                                          radius: float = 10,
                                          existing_image_ids: set = None,
                                          **kwargs) -> Tuple[List[List[StreetViewImage]], Dict[str, int]]:
        """
        Retrieve panoramas around multiple locations.
        Note: For Google Street View, this only returns the closest panorama per location.
        """
        print(f"Starting batch download around {len(locations)} locations...")
        
        # Use the at_locations method and convert to list of lists
        panos_list, stats = self.get_panos_at_locations_batched(
            locations, radius, existing_image_ids, **kwargs
        )
        
        # Convert single panoramas to lists
        result = []
        for pano in panos_list:
            if pano is not None:
                result.append([pano])
            else:
                result.append([])
        
        return result, stats
    
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great-circle distance between two points on Earth."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # Distance in meters
    
    def _fetch_panorama(self, pano_id: str):
        """Fetch and assemble a panorama from tiles (from download.py)"""
        
        def _fetch_tile(x, y, zoom=5):
            url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
            try:
                response = self._tile_session.get(url, timeout=20)
                if response.status_code == 200:
                    return x, y, Image.open(io.BytesIO(response.content))
                return x, y, None
            except Exception as e:
                if self.verbose:
                    print(f"Error fetching tile for pano {pano_id}, x={x}, y={y}: {e}")
                return x, y, None

        def _is_black_tile(tile):
            if tile is None:
                return True
            tile_array = np.array(tile)
            return np.all(tile_array == 0)

        def _find_panorama_dimensions():
            tiles_cache = {}
            x, y = 4, 1
            is_first = True
            while True:
                tile_info = _fetch_tile(x, y)
                if tile_info is None:
                    return None
                tile = tile_info[2]
                if tile is None:
                    return None
                if is_first:
                    is_first = False
                    if _is_black_tile(tile):
                        return None
                tiles_cache[(x, y)] = tile
                if _is_black_tile(tile):
                    y = y - 1
                    while True:
                        tile_info = _fetch_tile(x, y)
                        if tile_info is None:
                            return None
                        tile = tile_info[2]
                        tiles_cache[(x, y)] = tile
                        if _is_black_tile(tile):
                            return x - 1, y, tiles_cache
                        x += 1
                x += 1
                y += 1

        def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
            tiles_cache = existing_tiles.copy()
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                for x in range(max_x + 1):
                    for y in range(max_y + 1):
                        if (x, y) not in tiles_cache:
                            futures.append(executor.submit(_fetch_tile, x, y))
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        x, y, tile = result
                        if tile is not None:
                            tiles_cache[(x, y)] = tile

            # check for missing tiles — any gaps mean corrupted output
            expected = {(x, y) for x in range(max_x + 1) for y in range(max_y + 1)}
            missing = expected - set(tiles_cache.keys())
            if missing:
                if self.verbose:
                    print(f"Pano {pano_id}: {len(missing)} tiles failed after retries, skipping")
                return None
            return tiles_cache

        def _assemble_panorama(tiles, max_x, max_y):
            if not tiles:
                return None
            tile_size = list(tiles.values())[0].size[0]
            panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))
            for (x, y), tile in tiles.items():
                panorama.paste(tile, (x * tile_size, y * tile_size))
            return panorama

        def _crop(image):
            img_array = np.array(image)
            y_nonzero, x_nonzero, _ = np.nonzero(img_array)
            if y_nonzero.size > 0 and x_nonzero.size > 0:
                return img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
            return img_array

        dimension_result = _find_panorama_dimensions()
        if dimension_result is None:
            return None
        max_x, max_y, initial_tiles = dimension_result
        full_tiles = _fetch_remaining_tiles(max_x, max_y, initial_tiles)
        if full_tiles is None:
            return None
        assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
        if assembled_panorama is None:
            return None
        cropped_panorama = _crop(assembled_panorama)
        height, width = cropped_panorama.shape[:2]

        max_width = height * 2
        cropped_panorama = cropped_panorama[:, :max_width]
        
        resized = cv2.resize(cropped_panorama, (13312, 6656), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


# Example usage:
# gsv = GoogleStreetView(verbose=True)

# # Single panorama by ID
# svi = gsv.get_pano_with_id("some_pano_id")
# if svi:
#     with open('test_google.jpg', 'wb') as f:
#         f.write(svi.image_data)

# # Single location
# svi, stats = gsv.get_pano_at_location(47.6569533, -122.3132275, 50)
# if svi:
#     with open('test_location.jpg', 'wb') as f:
#         f.write(svi.image_data)
# 
# # Batch processing
# locations = [
#     (47.6569533, -122.3132275),
#     (47.6580113, -122.3147812),
#     (47.655801424438664, -122.30706602847431),
# ]
# results, stats = gsv.get_panos_at_locations_batched(locations, radius=50)