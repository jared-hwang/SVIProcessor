import pygeohash as pgh
import uuid
from typing import Tuple, Optional

class GeoPointID:
    def __init__(self, precision: int = 9):
        """
        Initialize GeoPointID encoder/decoder
        
        Args:
            precision: Geohash precision (default 9)
                      5 = ±2.4km, 6 = ±610m, 7 = ±76m, 8 = ±19m, 9 = ±2.4m
        """
        self.precision = precision
        self.points = {}  # Store mapping for reverse lookup if needed
    
    def encode(self, lat: float, lon: float, short_uuid: bool = True) -> str:
        """
        Create unique ID from lat/lon with geohash and UUID
        
        Args:
            lat: Latitude
            lon: Longitude
            short_uuid: If True, use only first 8 chars of UUID
        
        Returns:
            Unique ID string like 'geohash_uuid'
        """
        # Generate geohash
        geohash = pgh.encode(lat, lon, precision=self.precision)
        
        # Generate UUID (short version for readability if desired)
        unique_id = str(uuid.uuid4())
        if short_uuid:
            unique_id = unique_id[:8]
        
        # Combine with underscore separator
        full_id = f"{geohash}_{unique_id}"
        
        # Optionally store for reverse lookup
        self.points[full_id] = (lat, lon)
        
        return full_id
    
    def decode(self, geo_id: str) -> Tuple[float, float, str]:
        """
        Decode ID back to lat/lon and UUID
        
        Args:
            geo_id: The encoded geo ID
        
        Returns:
            Tuple of (latitude, longitude, uuid_part)
        """
        parts = geo_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid geo ID format: {geo_id}")
        
        geohash_part, uuid_part = parts
        
        # Decode geohash to lat/lon
        lat, lon = pgh.decode(geohash_part)
        
        return lat, lon, uuid_part
    
    def get_geohash_only(self, geo_id: str) -> str:
        """Extract just the geohash portion"""
        return geo_id.split('_')[0]
    
    def find_nearby(self, geo_id: str, stored_ids: list) -> list:
        """
        Find IDs that are geographically nearby
        
        Args:
            geo_id: Reference ID
            stored_ids: List of all stored IDs
        
        Returns:
            List of nearby IDs (sharing geohash prefix)
        """
        reference_hash = self.get_geohash_only(geo_id)
        
        # Check different prefix lengths for proximity
        nearby = []
        for check_id in stored_ids:
            if check_id == geo_id:
                continue
            check_hash = self.get_geohash_only(check_id)
            
            # Count matching prefix characters
            match_len = 0
            for i in range(min(len(reference_hash), len(check_hash))):
                if reference_hash[i] == check_hash[i]:
                    match_len += 1
                else:
                    break
            
            # Consider nearby if sharing at least 5 characters (±2.4km)
            if match_len >= 5:
                nearby.append((check_id, match_len))
        
        # Sort by proximity (more matching chars = closer)
        nearby.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in nearby]


# # Example usage
# def demo():
#     # Initialize encoder
#     geo_encoder = GeoPointID(precision=9)
    
#     # Sample points (some with same location)
#     points = [
#         (37.7749, -122.4194),  # San Francisco
#         (37.7749, -122.4194),  # Same location, different entity
#         (37.7750, -122.4195),  # Very close by
#         (37.8049, -122.4494),  # Few km away
#         (40.7128, -74.0060),   # New York (far away)
#     ]
    
#     # Encode all points
#     encoded_ids = []
#     for lat, lon in points:
#         geo_id = geo_encoder.encode(lat, lon)
#         encoded_ids.append(geo_id)
#         print(f"({lat:.4f}, {lon:.4f}) -> {geo_id}")
    
#     print("\n--- Decoding ---")
#     for geo_id in encoded_ids[:2]:
#         lat, lon, uuid_part = geo_encoder.decode(geo_id)
#         print(f"{geo_id} -> ({lat:.4f}, {lon:.4f}), UUID: {uuid_part}")
    
#     print("\n--- Finding Nearby Points ---")
#     reference = encoded_ids[0]
#     nearby = geo_encoder.find_nearby(reference, encoded_ids)
#     print(f"Points near {reference}:")
#     for near_id in nearby:
#         print(f"  {near_id}")


# # Alternative: Compact class for simple use cases
# class SimpleGeoID:
#     """Simpler version with just the essentials"""
    
#     @staticmethod
#     def encode(lat: float, lon: float, precision: int = 9) -> str:
#         """Create unique geo ID"""
#         geohash = pgh.encode(lat, lon, precision=precision)
#         short_uuid = str(uuid.uuid4())[:8]
#         return f"{geohash}_{short_uuid}"
    
#     @staticmethod
#     def decode(geo_id: str) -> Tuple[float, float]:
#         """Get approximate lat/lon from geo ID"""
#         geohash = geo_id.split('_')[0]
#         return pgh.decode(geohash)


# if __name__ == "__main__":
#     demo()
    
#     # Simple usage
#     print("\n--- Simple Usage ---")
#     geo_id = SimpleGeoID.encode(37.7749, -122.4194)
#     print(f"Encoded: {geo_id}")
#     lat, lon = SimpleGeoID.decode(geo_id)
#     print(f"Decoded: ({lat:.4f}, {lon:.4f})")