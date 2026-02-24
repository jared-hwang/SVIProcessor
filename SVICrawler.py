import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from typing import Optional, Callable
from sklearn.cluster import DBSCAN
from .CoordGeoHasher import GeoPointID
from .StreetViewService import StreetViewService, StreetViewImage
from .MapillaryStreetView import MapillaryStreetView

class SVICrawler():
    def __init__(self, project_name, project_workdir, bounding_topleft_latlong, bounding_botright_latlong):
        self.project_name = project_name
        self.project_workdir = project_workdir        

        # Create project directory if it doesn't exist
        self.project_dir = os.path.join(self.project_workdir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)

        self.bounding_topleft_latlong = bounding_topleft_latlong
        self.bounding_botright_latlong = bounding_botright_latlong
        self.geo_encoder = GeoPointID(precision=9)
        
        # Initialize dataframes
        self.sample_points_df = None
        self.sample_points_filepath = os.path.join(self.project_dir, 'sample_points')
        
        # Service-specific dataframes
        self.image_metadata_dfs = {}  # {service_name: dataframe}
        self.associations_dfs = {}     # {service_name: dataframe}
        
        # Try to load existing sample points
        self._try_load_existing_sample_points()

    def _try_load_existing_sample_points(self):
        """
        Attempt to load existing sample points from disk.
        Checks for different file formats in order of preference.
        """
        formats_to_try = ['parquet', 'csv', 'geojson', 'shapefile']
        
        for format in formats_to_try:
            if format == 'shapefile':
                filepath = f"{self.sample_points_filepath}.shp"
            else:
                filepath = f"{self.sample_points_filepath}.{format}"
            
            if os.path.exists(filepath):
                try:
                    print(f"Found existing sample points file: {filepath}")
                    self.load_dataframe('sample_points', format=format)
                    print(f"Successfully loaded {len(self.sample_points_df)} existing sample points")
                    return True
                except Exception as e:
                    print(f"Failed to load sample points from {filepath}: {e}")
                    continue
        
        return False
    
    def _create_point(self, lat, lon, **kwargs):
        """Helper to create a point dict with defaults"""
        point = {
            'sample_point_id': self.geo_encoder.encode(lat, lon),
            'sample_lat': lat,
            'sample_lon': lon,
            'geometry': Point(lon, lat),  # Add geometry for spatial operations
            'is_node': False,
            'is_100m': False,
            'is_tripoint': False,
            'node_id': None,
            'edge_u': None,
            'edge_v': None,
            'edge_key': None,
            'distance_from_u': None,
            'fraction_along_edge': None,
            'uvkey': None
        }
        point.update(kwargs)
        return point

    def generate_sample_points_across_area(self, interval_meters=20, include_intersections=True, 
                            network_type='all', custom_filter=None, verbose=False, 
                            force_regenerate=False, min_spacing_meters=None):
        """
        Generate sample points along OSM street network edges within the bounding box.
        Creates points at regular intervals along roads and optionally at intersections.
        Long edges (> interval_meters) are sampled at the specified interval, while
        short edges get a single midpoint sample.
        
        If sample points already exist on disk and force_regenerate=False, they will be
        loaded instead of regenerated.
        
        Parameters
        ----------
        interval_meters : float, default=20
            Distance interval for sampling points along edges.
        include_intersections : bool, default=True
            Whether to include network nodes (intersections) as sample points.
        network_type : str, default='all'
            OSMnx network type ('all', 'drive', 'walk', 'bike', etc.).
            Ignored if custom_filter is provided.
        custom_filter : str, optional
            Custom OSM filter string. If provided, overrides network_type.
            Mutually exclusive with network_type.
        force_regenerate : bool, default=False
            If True, regenerate sample points even if they exist on disk.
        min_spacing_meters : float, optional
            If provided, post-filter points so that no two points are closer than
            this distance (in meters). Uses a greedy algorithm that prioritizes
            keeping node/intersection points over edge sample points.
        
        Returns
        -------
        pd.DataFrame
            Sample points with columns: sample_point_id, sample_lat, sample_lon, 
            geometry, is_node, is_100m, is_tripoint, and additional metadata.
        """
        # Check if sample points already exist and should be used
        if not force_regenerate and self.sample_points_df is not None:
            print(f"Sample points already loaded ({len(self.sample_points_df)} points). "
                  "Use force_regenerate=True to regenerate them.")
            return self.sample_points_df
        
        # If we need to generate, proceed with original generation logic
        if force_regenerate:
            print("Force regenerating sample points...")
        else:
            print("No existing sample points found. Generating new ones...")
        
        # Update print statement to show which parameter is actually being used
        if custom_filter:
            print(f"Generating sample points: interval={interval_meters}m, "
                f"custom_filter='{custom_filter}', include_intersections={include_intersections}")
        else:
            print(f"Generating sample points: interval={interval_meters}m, "
                f"network_type='{network_type}', include_intersections={include_intersections}")
        
        topleft_lat, topleft_lon = self.bounding_topleft_latlong
        botright_lat, botright_lon = self.bounding_botright_latlong

        if custom_filter:
            G = ox.graph_from_bbox((topleft_lon, botright_lat, botright_lon, topleft_lat), 
                                    custom_filter=custom_filter, 
                                    retain_all=False)
        else:
            G = ox.graph_from_bbox((topleft_lon, botright_lat, botright_lon, topleft_lat), 
                                network_type=network_type,
                                retain_all=False)
        
        fig, ax = ox.plot_graph(G, 
                        save=True,        # Set to True to save the file
                        filepath='graphpng.png', # Specify the filename and extension
                        show=False,       # Set to False to prevent displaying
                        close=True,       # Set to True to close the figure
                        )

        
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        edges_gdf = ox.graph_to_gdfs(G, nodes=False)
        
        sample_points = []
        processed_nodes = set()
    
        
        def add_node(node_id):
            """Add a node if not already processed"""
            if node_id not in processed_nodes and node_id in nodes_gdf.index:
                node = nodes_gdf.loc[node_id]
                sample_points.append(self._create_point(
                    node.y, node.x,
                    is_node=True,
                    node_id=node_id
                ))
                processed_nodes.add(node_id)
        
        num_nodes, num_long_edges, num_long_edge_points = 0, 0, 0
        num_short_edges, num_short_edge_points = 0, 0

        # Process each edge with its associated points
        for idx, edge in edges_gdf.iterrows():
            u, v = idx[0], idx[1]
            edge_key = idx[2] if len(idx) > 2 else 0
            
            # Add start node
            if include_intersections:
                if u not in processed_nodes:
                    num_nodes += 1
                add_node(u)
            
            # Add edge sampling points
            line_geom = edge['geometry']
            edge_length = edge['length']
            edge_data = {'edge_u': u, 'edge_v': v, 'edge_key': edge_key}
            
            if edge_length > interval_meters:
                num_long_edges += 1
                # Long edge: sample at intervals
                num_intervals = int(edge_length // interval_meters)
                for i in range(1, num_intervals + 1):
                    distance = i * interval_meters
                    point = line_geom.interpolate(distance / edge_length, normalized=True)
                    num_long_edge_points += 1
                    sample_points.append(self._create_point(
                        point.y, point.x,
                        is_100m=True,
                        distance_from_u=distance,
                        **edge_data
                    ))
            else:
                # Short edge: sample at midpoint
                num_short_edges += 1
                point = line_geom.interpolate(0.5, normalized=True)
                num_short_edge_points += 1
                sample_points.append(self._create_point(
                    point.y, point.x,
                    is_tripoint=True,
                    distance_from_u=edge_length * 0.5,
                    fraction_along_edge=0.5,
                    uvkey=str((u, v, edge_key)),
                    **edge_data
                ))
            
            # Add end node
            if include_intersections:
                if v not in processed_nodes:
                    num_nodes += 1
                add_node(v)
        
        # Add any isolated nodes
        for idx in nodes_gdf.index:
            if idx not in processed_nodes:
                num_nodes += 1
                add_node(idx)
        
        # Convert to GeoDataFrame
        self.sample_points_df = gpd.GeoDataFrame(sample_points, crs='EPSG:4326')
        
        # Set index to sample_point_id for efficient querying
        self.sample_points_df.set_index('sample_point_id', inplace=True)
        
        print(f"Sampling complete: {len(self.sample_points_df)} total points")
        print(f"  - Nodes: {len(self.sample_points_df[self.sample_points_df['is_node']])}")
        print(f"  - Long edges (>{interval_meters}m): {num_long_edges} edges, {num_long_edge_points} points sampled")
        print(f"  - Short edges (<={interval_meters}m): {num_short_edges} edges, {num_short_edge_points} points sampled")
        
        # Apply minimum spacing filter if requested
        if min_spacing_meters is not None and min_spacing_meters > 0:
            pre_filter_count = len(self.sample_points_df)
            self.sample_points_df = self._enforce_min_spacing(self.sample_points_df, min_spacing_meters)
            post_filter_count = len(self.sample_points_df)
            print(f"  - Min spacing filter ({min_spacing_meters}m): {pre_filter_count} → {post_filter_count} points "
                  f"({pre_filter_count - post_filter_count} removed)")
        
        return self.sample_points_df

    def _enforce_min_spacing(self, gdf, min_spacing_meters):
        """
        Greedy filter to ensure no two points are closer than min_spacing_meters.
        
        Builds a single BallTree upfront and queries all neighbors within the
        minimum radius at once, then greedily selects points in priority order,
        excluding neighbors of already-kept points via a set.
        
        Points are prioritized: nodes (intersections) are kept first, then 
        long-edge sample points (is_100m), then short-edge midpoints (is_tripoint).
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with 'sample_lat', 'sample_lon', 'is_node', 'is_100m', 
            'is_tripoint' columns.
        min_spacing_meters : float
            Minimum distance in meters between any two kept points.
        
        Returns
        -------
        gpd.GeoDataFrame
            Filtered GeoDataFrame with minimum spacing enforced.
        """
        from sklearn.neighbors import BallTree
        
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        if len(gdf) <= 1:
            return gdf
        
        # Assign priority: nodes first, then long-edge points, then short-edge midpoints
        gdf = gdf.copy()
        gdf['_priority'] = np.where(
            gdf.get('is_node', False).astype(bool), 0,
            np.where(gdf.get('is_100m', False).astype(bool), 1, 2)
        )
        
        # Sort by priority so higher-priority points are processed first
        gdf = gdf.sort_values('_priority')
        
        # Build BallTree once from all coordinates in radians
        coords_rad = np.radians(gdf[['sample_lat', 'sample_lon']].values)
        tree = BallTree(coords_rad, metric='haversine')
        
        # Convert min spacing to radians (Earth radius ≈ 6371000 m)
        min_spacing_rad = min_spacing_meters / 6_371_000.0
        
        # Query all neighbors within min_spacing for every point at once
        print(f"  Building spatial index for {len(gdf)} points...")
        all_neighbors = tree.query_radius(coords_rad, r=min_spacing_rad)
        
        # Greedy selection using positional indices
        excluded = set()
        kept_positional = []
        
        iterator = range(len(gdf))
        if has_tqdm:
            iterator = tqdm(iterator, desc="  Enforcing min spacing", unit="pts")
        
        for i in iterator:
            if i in excluded:
                continue
            
            kept_positional.append(i)
            
            # Exclude all neighbors of this point (except itself)
            for neighbor_idx in all_neighbors[i]:
                if neighbor_idx != i:
                    excluded.add(neighbor_idx)
        
        # Map positional indices back to dataframe index
        kept_df_indices = gdf.index[kept_positional]
        result = gdf.loc[kept_df_indices].drop(columns=['_priority'])
        return result

    def generate_sample_points_from_latlongs(self, lat_longs, network_type='all', 
                                            custom_filter=None, buffer=0.005, 
                                            max_cluster_span=0.05, force_regenerate=False,
                                            verbose=False):
        """
        Snap lat/long points to the nearest point on an OSM network edge and create
        point objects with edge details. Automatically clusters points to minimize
        OSM graph downloads.
        
        Parameters:
        -----------
        lat_longs : list of tuples
            List of (latitude, longitude) tuples
        network_type : str, default='all'
            OSMnx network type ('all', 'drive', 'walk', 'bike', etc.).
            Ignored if custom_filter is provided.
        custom_filter : str, optional
            Custom OSM filter string. If provided, overrides network_type.
            Mutually exclusive with network_type.
        buffer : float
            Buffer in degrees to add around bounding box
        max_cluster_span : float
            Maximum span in degrees for a single cluster (controls graph size)
        force_regenerate : bool, default=False
            If True, regenerate sample points even if they exist on disk.
        verbose : bool, default=False
            If True, print detailed progress information
        
        Returns:
        --------
        pd.DataFrame or None
            Sample points with columns: sample_point_id, sample_lat, sample_lon, 
            geometry, and additional edge metadata. Returns None if generation fails.
        """
        
        # Check if sample points already exist and should be used
        if not force_regenerate and self.sample_points_df is not None:
            print(f"Sample points already loaded ({len(self.sample_points_df)} points). "
                "Use force_regenerate=True to regenerate them.")
            return self.sample_points_df
        
        # If we need to generate, proceed with original generation logic
        if force_regenerate:
            print("Force regenerating sample points from lat/longs...")
        else:
            print("No existing sample points found. Generating new ones from lat/longs...")
        
        # Update print statement to show which parameter is actually being used
        if custom_filter:
            print(f"Snapping {len(lat_longs)} points to network: "
                f"custom_filter='{custom_filter}', buffer={buffer}, max_cluster_span={max_cluster_span}")
        else:
            print(f"Snapping {len(lat_longs)} points to network: "
                f"network_type='{network_type}', buffer={buffer}, max_cluster_span={max_cluster_span}")
        
        if not lat_longs:
            print("Error: No lat/long points provided")
            self.sample_points_df = None
            return None
        
        # Cluster points geographically
        clusters = self._cluster_points_geographically(lat_longs, max_cluster_span)
        print(f"Created {len(clusters)} cluster(s) for processing")
        
        all_snapped_points = []
        successful_clusters = 0
        failed_clusters = 0
        
        # Process each cluster separately
        for cluster_id, cluster_points in clusters.items():
            if verbose:
                print(f"  Processing cluster {cluster_id + 1}/{len(clusters)} with {len(cluster_points)} points...")
            
            # Get points for this cluster
            cluster_lats = [lat for lat, lon in cluster_points]
            cluster_lons = [lon for lat, lon in cluster_points]


            # G = ox.graph_from_bbox((topleft_lon, botright_lat, botright_lon, topleft_lat), 
            #                         custom_filter=custom_filter, 
            #                         retain_all=False)

            # (min(cluster_lons)+buffer,min(cluster_lats)+buffer,max(cluster_lons)+buffer,max(cluster_lats)+buffer)
            # Create graph for this cluster
            try:
                if custom_filter:
                    G = ox.graph_from_bbox(
                        (min(cluster_lons)-buffer,min(cluster_lats)-buffer,max(cluster_lons)+buffer,max(cluster_lats)+buffer),
                        custom_filter=custom_filter)
                else:
                    G = ox.graph_from_bbox(                        
                        (min(cluster_lons)-buffer,min(cluster_lats)-buffer,max(cluster_lons)+buffer,max(cluster_lats)+buffer),
                        network_type=network_type)

                # Process points in this cluster
                cluster_snapped = self._snap_points_to_graph(G, cluster_points)
                all_snapped_points.extend(cluster_snapped)
                successful_clusters += 1
                
                # Create the plot
                # fig, ax = ox.plot_graph(G, show=False, close=False)
                # for point_i in cluster_snapped:
                #     # Add the first point as a red dot
                #     ax.scatter(point_i['sample_lon'], point_i['sample_lat'], c='red', s=100, zorder=5, edgecolors='black', linewidths=1)
                #     # Add the second point as a blue dot  
                #     ax.scatter(point_i['original_lon'], point_i['original_lat'], c='blue', s=100, zorder=5, edgecolors='black', linewidths=1)
                # # plt.savefig('test.png')
                # plt.show()

                if verbose:
                    print(f"    Successfully snapped {len(cluster_snapped)} points")
                
            except Exception as e:
                print(f"  Error processing cluster {cluster_id}: {e}")
                failed_clusters += 1
                continue
        
        # Convert to GeoDataFrame
        if all_snapped_points:
            # Extract data from point dictionaries
            data_for_df = []
            for point in all_snapped_points:
                # Assuming point is a dictionary with the required fields
                # You may need to adjust based on your _create_point method
                data_for_df.append(point)
            
            self.sample_points_df = gpd.GeoDataFrame(data_for_df, crs='EPSG:4326')
            
            # Set index to sample_point_id for efficient querying
            if 'sample_point_id' in self.sample_points_df.columns:
                self.sample_points_df.set_index('sample_point_id', inplace=True)
        else:
            # Return None instead of empty GeoDataFrame
            print("Error: No points could be successfully snapped to the network")
            self.sample_points_df = None
            return None
        
        # Print summary statistics
        print(f"Snapping complete: {len(self.sample_points_df)} points successfully snapped")
        print(f"  - Clusters processed: {successful_clusters} successful, {failed_clusters} failed")
        if len(self.sample_points_df) > 0:
            # Count different point types if columns exist
            if 'is_node' in self.sample_points_df.columns:
                num_nodes = len(self.sample_points_df[self.sample_points_df['is_node']])
                print(f"  - Node points: {num_nodes}")
            
            # Show distance statistics if available
            if 'distance_from_u' in self.sample_points_df.columns:
                avg_distance = self.sample_points_df['distance_from_u'].mean()
                print(f"  - Average distance from u node: {avg_distance:.2f}m")
            
            # Show snapping accuracy if original coordinates were preserved
            if 'original_lat' in self.sample_points_df.columns and 'original_lon' in self.sample_points_df.columns:
                # Calculate snapping distances
                from geopy.distance import geodesic
                snapping_distances = []
                for idx, row in self.sample_points_df.iterrows():
                    if pd.notna(row.get('original_lat')) and pd.notna(row.get('original_lon')):
                        original = (row['original_lat'], row['original_lon'])
                        snapped = (row['sample_lat'], row['sample_lon'])
                        distance = geodesic(original, snapped).meters
                        snapping_distances.append(distance)
                
                if snapping_distances:
                    avg_snap_distance = np.mean(snapping_distances)
                    max_snap_distance = np.max(snapping_distances)
                    print(f"  - Snapping accuracy: avg={avg_snap_distance:.2f}m, max={max_snap_distance:.2f}m")
        
        return self.sample_points_df

    def _cluster_points_geographically(self, lat_longs, max_cluster_span=0.05):
        """
        Cluster points based on geographic proximity using DBSCAN.
        
        Parameters:
        -----------
        lat_longs : list of tuples
            List of (latitude, longitude) tuples
        max_cluster_span : float
            Maximum distance in degrees between points in a cluster
        
        Returns:
        --------
        dict mapping cluster_id to list of (lat, lon) tuples
        """
        
        # Convert to numpy array for clustering
        coords = np.array(lat_longs)
        
        # Use DBSCAN with epsilon = max_cluster_span
        # This ensures no two points in a cluster are farther than eps apart
        clustering = DBSCAN(eps=max_cluster_span, min_samples=1, metric='euclidean').fit(coords)
        
        # Group points by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            clusters[label].append(lat_longs[i])
        
        return dict(clusters)

    def _snap_points_to_graph(self, G, lat_longs):
        """
        Snap a set of lat/long points to the given graph.
        """
        # Convert to x, y format
        points = [(lon, lat) for lat, lon in lat_longs]
        
        # Find nearest edges
        edges, distances = ox.nearest_edges(
            G, 
            X=[p[0] for p in points],
            Y=[p[1] for p in points],
            return_dist=True
        )
        
        snapped_points = []
        
        for i, (u, v, key) in enumerate(edges):
            edge_data = G.edges[u, v, key]
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            
            # Get edge geometry
            if 'geometry' in edge_data:
                edge_geom = edge_data['geometry']
            else:
                edge_geom = LineString([(u_node['x'], u_node['y']), 
                                    (v_node['x'], v_node['y'])])
            
            # Find nearest point on edge
            original_point = Point(points[i])
            nearest_point = edge_geom.interpolate(edge_geom.project(original_point))
            
            # Calculate distance from u node to snapped point
            distance_along_edge = edge_geom.project(nearest_point)
            total_edge_length = edge_geom.length
            
            # Calculate fraction along edge
            fraction_along_edge = distance_along_edge / total_edge_length if total_edge_length > 0 else 0
            
            # Calculate actual distance in meters
            u_coords = (u_node['y'], u_node['x'])
            snapped_coords = (nearest_point.y, nearest_point.x)
            distance_from_u_meters = geodesic(u_coords, snapped_coords).meters
            
            # Calculate snapping distance for logging
            original_lat, original_lon = lat_longs[i]
            snapping_distance = distances[i] if distances is not None else None
            
            # Create the point using _create_point for consistency
            point = self._create_point(
                lat=nearest_point.y,
                lon=nearest_point.x,
                is_node=False, 
                is_100m=False,
                is_tripoint=False,
                edge_u=u,
                edge_v=v,
                edge_key=key,
                distance_from_u=distance_from_u_meters,
                fraction_along_edge=fraction_along_edge,
                uvkey=f"({u}, {v}, {key})",
                original_lat=original_lat,  # Store original coordinates for accuracy calculation
                original_lon=original_lon,
                snapping_distance=snapping_distance
            )
            
            snapped_points.append(point)
            
        return snapped_points

    def save_dataframe(self, df_name, format='parquet', df=None, service_name=None):
        """
        Save a dataframe to disk in various formats.
        
        Parameters
        ----------
        df_name : str
            Name of the dataframe to save. Can be:
            - 'sample_points': Save the sample points geodataframe
            - 'image_metadata': Save the image metadata dataframe (requires service_name)
            - 'associations': Save the associations dataframe (requires service_name)
            - Any custom name if df is provided
        format : str, default='parquet'
            File format ('parquet', 'csv', 'geojson', 'shapefile')
            Note: geojson/shapefile only work with GeoDataFrames
        df : pd.DataFrame or gpd.GeoDataFrame, optional
            Custom dataframe to save. If None, will look for predefined dataframes
        service_name : str, optional
            Name of the service (required for image_metadata and associations)
        """
        # Determine which dataframe to save
        if df is not None:
            dataframe = df
            filepath_base = os.path.join(self.project_dir, df_name)
        elif df_name == 'sample_points':
            dataframe = self.sample_points_df
            filepath_base = self.sample_points_filepath
        elif df_name == 'image_metadata':
            if not service_name:
                raise ValueError("service_name is required for saving image_metadata")
            if service_name not in self.image_metadata_dfs:
                raise ValueError(f"No image_metadata for service '{service_name}'")
            dataframe = self.image_metadata_dfs[service_name]
            service_dir = os.path.join(self.project_dir, service_name)
            os.makedirs(service_dir, exist_ok=True)
            filepath_base = os.path.join(service_dir, 'image_metadata')
        elif df_name == 'associations':
            if not service_name:
                raise ValueError("service_name is required for saving associations")
            if service_name not in self.associations_dfs:
                raise ValueError(f"No associations for service '{service_name}'")
            dataframe = self.associations_dfs[service_name]
            service_dir = os.path.join(self.project_dir, service_name)
            os.makedirs(service_dir, exist_ok=True)
            filepath_base = os.path.join(service_dir, 'sample_point_associations')
        else:
            raise ValueError(f"Unknown dataframe name: {df_name}. Use 'sample_points', 'image_metadata', 'associations', or provide a custom dataframe with df parameter.")
        
        if dataframe is None or (hasattr(dataframe, 'empty') and dataframe.empty):
            raise ValueError(f"No {df_name} to save. Dataframe is empty or None.")
        
        # Save based on format
        is_geodf = isinstance(dataframe, gpd.GeoDataFrame)
        
        if format == 'parquet':
            dataframe.to_parquet(f"{filepath_base}.parquet")
        elif format == 'csv':
            df_to_save = dataframe.copy()
            if is_geodf and 'geometry' in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=['geometry'])
            df_to_save.to_csv(f"{filepath_base}.csv")
        elif format == 'geojson':
            if not is_geodf:
                raise ValueError(f"GeoJSON format requires a GeoDataFrame, but {df_name} is a regular DataFrame")
            dataframe.to_file(f"{filepath_base}.geojson", driver='GeoJSON')
        elif format == 'shapefile':
            if not is_geodf:
                raise ValueError(f"Shapefile format requires a GeoDataFrame, but {df_name} is a regular DataFrame")
            dataframe.to_file(f"{filepath_base}.shp")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'parquet', 'csv', 'geojson', or 'shapefile'")
        
        print(f"{df_name} saved to {filepath_base}.{format if format != 'shapefile' else 'shp'}")

    def load_dataframe(self, df_name, format='parquet', service_name=None):
        """
        Load a dataframe from disk.
        
        Parameters
        ----------
        df_name : str
            Name of the dataframe to load. Can be:
            - 'sample_points': Load into self.sample_points_df
            - 'image_metadata': Load into self.image_metadata_dfs[service_name]
            - 'associations': Load into self.associations_dfs[service_name]
            - Any custom name: Returns the dataframe only
        format : str, default='parquet'
            File format to load from
        service_name : str, optional
            Name of the service (required for image_metadata and associations)
        
        Returns
        -------
        pd.DataFrame or gpd.GeoDataFrame
            The loaded dataframe (also sets instance variable for known dataframes)
        """
        # Determine filepath
        if df_name == 'sample_points':
            filepath_base = self.sample_points_filepath
        elif df_name == 'image_metadata':
            if not service_name:
                raise ValueError("service_name is required for loading image_metadata")
            service_dir = os.path.join(self.project_dir, service_name)
            filepath_base = os.path.join(service_dir, 'image_metadata')
        elif df_name == 'associations':
            if not service_name:
                raise ValueError("service_name is required for loading associations")
            service_dir = os.path.join(self.project_dir, service_name)
            filepath_base = os.path.join(service_dir, 'sample_point_associations')
        else:
            filepath_base = os.path.join(self.project_dir, df_name)
        
        # Load based on format
        if format == 'parquet':
            filepath = f"{filepath_base}.parquet"
            # Try to detect if it's a GeoDataFrame
            try:
                df = gpd.read_parquet(filepath)
            except:
                df = pd.read_parquet(filepath)
        elif format == 'csv':
            filepath = f"{filepath_base}.csv"
            df = pd.read_csv(filepath, index_col=0)
            # For sample_points, recreate geometry
            if df_name == 'sample_points' and 'sample_lat' in df.columns and 'sample_lon' in df.columns:
                geometry = [Point(row['sample_lon'], row['sample_lat']) for _, row in df.iterrows()]
                df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        elif format == 'geojson':
            filepath = f"{filepath_base}.geojson"
            df = gpd.read_file(filepath)
            if 'sample_point_id' in df.columns:
                df.set_index('sample_point_id', inplace=True)
        elif format == 'shapefile':
            filepath = f"{filepath_base}.shp"
            df = gpd.read_file(filepath)
            if 'sample_point_id' in df.columns:
                df.set_index('sample_point_id', inplace=True)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Loaded {len(df)} records from {filepath}")
        
        # Set instance variables for known dataframes
        if df_name == 'sample_points':
            self.sample_points_df = df
        elif df_name == 'image_metadata':
            self.image_metadata_dfs[service_name] = df
        elif df_name == 'associations':
            self.associations_dfs[service_name] = df
        
        return df

    def download_images_within_region(self, 
                                    svi_service: StreetViewService,
                                    sample_point_filter_func: Optional[Callable[[pd.Series], bool]] = None,
                                    image_filter_func: Optional[Callable[[StreetViewImage], bool]] = None,
                                    batch_size: int = 100,
                                    resume: bool = True,
                                    single_pano_per_point: bool = True,
                                    **kwargs) -> Dict:
        """
        Download street view images for sample points within the region.
        
        :param svi_service: StreetViewService instance
        :param sample_point_filter_func: Optional filter for sample points (takes pd.Series, returns bool)
        :param image_filter_func: Optional post-download filter for images
        :param batch_size: Number of sample points per batch
        :param resume: Resume from last batch if interrupted
        :param single_pano_per_point: If True, only closest pano; if False, all panos within radius. Default radius=10m
        :param kwargs: Arguments for svi_service methods (radius, min_date, max_date)
        :return: Dictionary with success metrics
        """
        if self.sample_points_df is None:
            raise ValueError("No sample points generated. Run generate_sample_points() first.")
        
        # Get service name from the service instance
        service_name = svi_service.name
        print(f"\nStarting download for service: {service_name}")
                
        # Initialize tracking columns (service-specific)
        self._init_download_tracking_columns(service_name)
        
        # Setup directories and file paths (service-specific)
        paths = self._setup_download_paths(service_name)
        os.makedirs(paths['images_dir'], exist_ok=True)
        
        # Load or initialize state
        state = self._load_download_state(paths, resume, service_name)
        
        # Get points to process
        points_to_process = self._get_points_for_download(
            sample_point_filter_func, 
            state['processed_points'] if resume else set(),
            service_name
        )
        
        if not points_to_process:
            print(f"No points to process after filtering for {service_name}.")
            return {'total_sample_points': len(self.sample_points_df), 'points_processed': 0}
        
        print(f"Points to process: {len(points_to_process)} out of {len(self.sample_points_df)} total")
        
        # Initialize statistics
        stats = self._init_download_stats(points_to_process, state['processed_points'], state['existing_image_ids'])
        stats['service_name'] = service_name
        
        # Process batches
        num_batches = (len(points_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(state['start_batch'], num_batches):
            batch_points = points_to_process[batch_num * batch_size:(batch_num + 1) * batch_size]
            print(f"\nProcessing batch {batch_num + 1}/{num_batches} ({len(batch_points)} points)")
            
            # Download and process batch
            batch_stats = self._process_download_batch(
                batch_points, svi_service, single_pano_per_point,
                image_filter_func, state, paths, service_name, **kwargs
            )
            
            # Update statistics
            stats['points_processed'] += len(batch_points)
            for key in ['points_with_svi', 'points_without_svi', 'total_images_downloaded', 
                        'download_failures', 'images_filtered_out']:
                stats[key] += batch_stats.get(key, 0)
            
            # Save progress
            self._save_download_state(state, paths, batch_num, num_batches, stats['points_processed'], service_name)
            self.save_dataframe('sample_points', format='parquet')
            self.save_dataframe('sample_points', format='csv')
            
            print(f"Batch complete. Points with SVI: {batch_stats['points_with_svi']}, "
                f"Images downloaded: {batch_stats['images_downloaded']}")
        
        # Finalize
        stats['unique_images'] = len(state['existing_image_ids'])
        self._save_json(paths['stats_file'], stats)
        
        # Clean up progress file if complete
        if batch_num == num_batches - 1 and os.path.exists(paths['progress_file']):
            os.remove(paths['progress_file'])
        
        self._print_download_summary(stats, sample_point_filter_func, image_filter_func, service_name)
        return stats

    def _init_download_tracking_columns(self, service_name: str):
        """Initialize SVI tracking columns in sample_points_df for download pipeline."""
        col_prefix = f"{service_name}_"
        defaults = {
            f'{col_prefix}has_svi': pd.NA,  # Use NA instead of False to indicate "not checked"
            f'{col_prefix}svi_count': 0, 
            f'{col_prefix}svi_image_ids': None
        }
        for col, default in defaults.items():
            if col not in self.sample_points_df.columns:
                self.sample_points_df[col] = default

    def _setup_download_paths(self, service_name: str) -> Dict[str, str]:
        """Setup and return all file paths for download pipeline."""
        service_dir = os.path.join(self.project_dir, service_name)
        return {
            'images_dir': os.path.join(service_dir, 'images'),
            'metadata_file': os.path.join(service_dir, 'image_metadata.csv'),
            'associations_file': os.path.join(service_dir, 'sample_point_associations.csv'),
            'progress_file': os.path.join(service_dir, 'download_progress.json'),
            'stats_file': os.path.join(service_dir, 'download_statistics.json')
        }

    def _load_download_state(self, paths: Dict[str, str], resume: bool, service_name: str) -> Dict:
        """Load or initialize download state for the download pipeline."""
        state = {
            'existing_image_ids': set(),
            'processed_points': set(),
            'start_batch': 0
        }
        
        # Initialize instance variables if not resuming
        if not resume:
            self.image_metadata_dfs[service_name] = pd.DataFrame()
            self.associations_dfs[service_name] = pd.DataFrame()
            if os.path.exists(paths['progress_file']):
                os.remove(paths['progress_file'])
            return state
        
        # Load progress
        if os.path.exists(paths['progress_file']):
            with open(paths['progress_file'], 'r') as f:
                progress = json.load(f)
                state['start_batch'] = progress.get('last_completed_batch', 0) + 1
                print(f"Resuming from batch {state['start_batch']}")
        
        # Load existing data into instance variables
        if os.path.exists(paths['metadata_file']):
            self.image_metadata_dfs[service_name] = pd.read_csv(paths['metadata_file'])
        else:
            self.image_metadata_dfs[service_name] = pd.DataFrame()
            
        if os.path.exists(paths['associations_file']):
            self.associations_dfs[service_name] = pd.read_csv(paths['associations_file'])
        else:
            self.associations_dfs[service_name] = pd.DataFrame()
        
        # Extract IDs from loaded data, removing the service prefix for checking
        if not self.image_metadata_dfs[service_name].empty:
            # Strip the service prefix when creating the existing_image_ids set
            prefix = f"{service_name}_"
            state['existing_image_ids'] = set(
                img_id.replace(prefix, '', 1) if img_id.startswith(prefix) else img_id 
                for img_id in self.image_metadata_dfs[service_name]['image_id'].values
            )
            print(f"Loaded {len(state['existing_image_ids'])} existing images for {service_name}")
        
        if not self.associations_dfs[service_name].empty:
            state['processed_points'] = set(self.associations_dfs[service_name]['sample_point_id'].unique())
            print(f"Loaded {len(state['processed_points'])} already processed points for {service_name}")
        
        return state

    def _get_points_for_download(self, filter_func: Optional[Callable], 
                            processed_points: set, service_name: str) -> List[str]:
        """Get filtered list of sample points to process for download."""
        points = []
        col_prefix = f"{service_name}_"
        has_svi_col = f'{col_prefix}has_svi'
        
        for point_id in self.sample_points_df.index:
            # Check if this point has been checked before
            # If has_svi column exists and is not null, it means we've checked this point
            if has_svi_col in self.sample_points_df.columns:
                has_svi_value = self.sample_points_df.loc[point_id, has_svi_col]
                # Skip if we've already checked this point (whether it had SVI or not)
                if pd.notna(has_svi_value):
                    continue
            
            # Apply custom filter if provided
            if filter_func and not filter_func(self.sample_points_df.loc[point_id]):
                continue
                
            points.append(point_id)
        return points

    def _process_download_batch(self, batch_points: List[str], svi_service: StreetViewService,
                    single_pano: bool, image_filter: Optional[Callable],
                    state: Dict, paths: Dict, service_name: str, **kwargs) -> Dict:
        """Process a single batch of points for downloading."""
        # Prepare locations
        locations = [(self.sample_points_df.loc[pid, 'sample_lat'], 
                    self.sample_points_df.loc[pid, 'sample_lon']) 
                    for pid in batch_points]
        
        # Pass existing_image_ids (which now contains non-prefixed IDs) to prevent re-downloading
        if single_pano:
            results, api_stats = svi_service.get_panos_at_locations_batched(
                locations, existing_image_ids=state['existing_image_ids'], **kwargs)
            # Convert single panos to lists for uniform processing
            # Now results maintain position, with None for missing panos
            results = [[p] if p else [] for p in results]
        else:
            results, api_stats = svi_service.get_panos_around_locations_batched(
                locations, existing_image_ids=state['existing_image_ids'], **kwargs)
        # Process results
        batch_stats = {'points_with_svi': 0, 'points_without_svi': 0, 
                    'images_downloaded': 0, 'download_failures': 0, 'images_filtered_out': 0}
        batch_metadata = []
        batch_associations = []
        
        for point_id, panos in zip(batch_points, results):
            # Filter images if needed
            if panos and image_filter:
                filtered = [p for p in panos if image_filter(p)]
                batch_stats['images_filtered_out'] += len(panos) - len(filtered)
                panos = filtered
            if not panos:
                self._update_point_download_status(point_id, False, 0, None, service_name)
                batch_stats['points_without_svi'] += 1
                continue
            
            # Process panoramas
            image_ids = []
            for pano in panos:
                # Create the prefixed image ID for storage
                prefixed_image_id = f"{service_name}_{pano.pano_id}"
                
                # Create association with prefixed ID
                association = {
                    'sample_point_id': point_id,
                    'image_id': prefixed_image_id,  # Use prefixed ID
                    'distance_meters': getattr(pano, 'dist_from_request', None)
                }
                batch_associations.append(association)
                
                if hasattr(pano, 'already_downloaded') and pano.already_downloaded:
                    image_ids.append(prefixed_image_id)  # Use prefixed ID
                    continue
                
                # Check if already downloaded using the original (non-prefixed) ID
                if pano.pano_id in state['existing_image_ids']:
                    image_ids.append(prefixed_image_id)
                    continue
                
                # Save new image with prefixed filename
                if self._save_downloaded_image(pano, paths['images_dir'], service_name):
                    batch_metadata.append({
                        'image_id': prefixed_image_id,  # Use prefixed ID
                        'filename': f"{prefixed_image_id}.jpg",  # Use prefixed filename
                        'lat': pano.lat,
                        'lon': pano.lon,
                        'heading': pano.heading,
                        'pitch': pano.pitch,
                        'fov': pano.fov,
                        'timestamp': pano.timestamp,
                        'download_timestamp': datetime.now().isoformat()
                    })
                    image_ids.append(prefixed_image_id)
                    # Add the original (non-prefixed) ID to existing set for future checks
                    state['existing_image_ids'].add(pano.pano_id)
                    batch_stats['images_downloaded'] += 1
                else:
                    batch_stats['download_failures'] += 1
            
            # Update point status
            if image_ids:
                self._update_point_download_status(point_id, True, len(image_ids), ','.join(image_ids), service_name)
                batch_stats['points_with_svi'] += 1
            else:
                self._update_point_download_status(point_id, False, 0, None, service_name)
                batch_stats['points_without_svi'] += 1
        
        # Update instance variables
        if batch_metadata:
            self.image_metadata_dfs[service_name] = pd.concat(
                [self.image_metadata_dfs[service_name], pd.DataFrame(batch_metadata)], 
                ignore_index=True
            )
        if batch_associations:
            self.associations_dfs[service_name] = pd.concat(
                [self.associations_dfs[service_name], pd.DataFrame(batch_associations)], 
                ignore_index=True
            )
        
        return batch_stats

    def _update_point_download_status(self, point_id: str, has_svi: bool, count: int, 
                                    image_ids: Optional[str], service_name: str):
        """Update sample point with SVI download information."""
        col_prefix = f"{service_name}_"
        self.sample_points_df.at[point_id, f'{col_prefix}has_svi'] = has_svi
        self.sample_points_df.at[point_id, f'{col_prefix}svi_count'] = count
        self.sample_points_df.at[point_id, f'{col_prefix}svi_image_ids'] = image_ids

    def _save_downloaded_image(self, pano: StreetViewImage, images_dir: str, service_name: str) -> bool:
        """Save downloaded panorama image to disk with service-prefixed filename."""
        try:
            prefixed_filename = f"{service_name}_{pano.pano_id}.jpg"
            image_path = os.path.join(images_dir, prefixed_filename)
            with open(image_path, 'wb') as f:
                f.write(pano.image_data)
            return True
        except Exception as e:
            print(f"Error saving image {service_name}_{pano.pano_id}: {e}")
            return False

    def _save_download_state(self, state: Dict, paths: Dict, batch_num: int, 
                num_batches: int, points_processed: int, service_name: str):
        """Save current download state for the download pipeline."""
        # Save instance variable dataframes
        if not self.image_metadata_dfs[service_name].empty:
            self.image_metadata_dfs[service_name].to_csv(paths['metadata_file'], index=False)
        
        if not self.associations_dfs[service_name].empty:
            self.associations_dfs[service_name].to_csv(paths['associations_file'], index=False)
        
        # Save progress
        self._save_json(paths['progress_file'], {
            'last_completed_batch': batch_num,
            'total_batches': num_batches,
            'points_processed': points_processed,
            'timestamp': datetime.now().isoformat()
        })

    def _save_json(self, filepath: str, data: Dict):
        """Save dictionary as JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _init_download_stats(self, points_to_process: List, processed_points: set, 
                existing_images: set) -> Dict:
        """Initialize statistics dictionary for download tracking."""
        return {
            'total_sample_points': len(self.sample_points_df),
            'points_filtered_out': len(self.sample_points_df) - len(points_to_process) - len(processed_points),
            'points_processed': len(processed_points),
            'points_with_svi': 0,
            'points_without_svi': 0,
            'total_images_downloaded': 0,
            'unique_images': len(existing_images),
            'download_failures': 0,
            'images_filtered_out': 0
        }

    def _print_download_summary(self, stats: Dict, sample_filter: Optional[Callable], 
                    image_filter: Optional[Callable], service_name: str):
        """Print download summary after completion."""
        print("\n" + "="*60)
        print(f"Download Complete for {service_name}!")
        print("="*60)
        for key, label in [
            ('total_sample_points', 'Total sample points'),
            ('points_filtered_out', 'Points filtered out'),
            ('points_processed', 'Points processed'),
            ('points_with_svi', 'Points with SVI'),
            ('points_without_svi', 'Points without SVI'),
            ('unique_images', 'Unique images downloaded'),
            ('total_images_downloaded', 'Total downloads'),
            ('download_failures', 'Download failures'),
            ('images_filtered_out', 'Images filtered out')
        ]:
            if key == 'points_filtered_out' and not sample_filter:
                continue
            if key == 'images_filtered_out' and not image_filter:
                continue
            print(f"{label}: {stats[key]}")

    def run_func_on_svis(self, 
                        service_name: str,
                        run_name: str,
                        run_func: Callable[[pd.DataFrame], Dict[str, Dict]],
                        filter_func: Optional[Callable[[pd.Series], bool]] = None,
                        batch_size: int = 20,
                        resume: bool = True) -> Dict[str, Dict]:
        """
        Run a function on street view images for a given service, with batching and resume capability.
        
        Parameters
        ----------
        service_name : str
            Name of the service whose images to process
        run_name : str
            Unique identifier for this run (used for saving/resuming progress)
        run_func : Callable[[pd.DataFrame], Dict[str, Dict]]
            Function to run on image data. 
            - If batch_size=1: receives a single-row DataFrame, returns Dict
            - If batch_size>1: receives a DataFrame with multiple rows, returns Dict[str, Dict] 
            where keys are image_ids
        filter_func : Optional[Callable[[pd.Series], bool]]
            Optional filter to apply before running the function
        batch_size : int, default=1
            Number of rows to process at once
        resume : bool, default=True
            Whether to resume from previous run if it exists
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary where keys are image_ids and values are the returned dicts from run_func
        """
        # Setup paths for this run
        service_dir = os.path.join(self.project_dir, service_name)
        runs_dir = os.path.join(service_dir, 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        
        results_file = os.path.join(runs_dir, f'{run_name}_results.json')
        progress_file = os.path.join(runs_dir, f'{run_name}_progress.json')
        
        # Check if we have the dataframe in memory
        if service_name not in self.image_metadata_dfs or self.image_metadata_dfs[service_name].empty:
            # Try to load from disk
            try:
                print(f"Loading image metadata for service: {service_name}")
                self.load_dataframe('image_metadata', format='parquet', service_name=service_name)
            except FileNotFoundError:
                # Try CSV as fallback
                try:
                    self.load_dataframe('image_metadata', format='csv', service_name=service_name)
                except FileNotFoundError:
                    raise ValueError(f"No image metadata found for service '{service_name}'. "
                                "Please run download_images_within_region first.")
        
        # Get the dataframe
        df = self.image_metadata_dfs[service_name]
        
        if df.empty:
            print(f"No images to process for service {service_name}")
            return {}
        
        # Apply filter if provided
        if filter_func:
            mask = df.apply(filter_func, axis=1)
            filtered_df = df[mask]
            print(f"Processing {len(filtered_df)} images out of {len(df)} after filtering")
        else:
            filtered_df = df
            print(f"Processing all {len(filtered_df)} images")
        
        # Load existing results if resuming
        results = {}
        processed_image_ids = set()
        start_batch = 0
        
        if resume and os.path.exists(results_file):
            print(f"Found existing results for run '{run_name}', loading...")
            with open(results_file, 'r') as f:
                results = json.load(f)
                processed_image_ids = set(results.keys())
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_batch = progress.get('last_completed_batch', 0) + 1
            print(f"Resuming from batch {start_batch}, {len(processed_image_ids)} images already processed")
        elif not resume and os.path.exists(results_file):
            # If not resuming but file exists, back it up
            backup_file = f"{results_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(results_file, backup_file)
            print(f"Existing results backed up to: {backup_file}")
        
        # Filter out already processed images if resuming
        if processed_image_ids:
            # Keep only rows that haven't been processed
            filtered_df = filtered_df[~filtered_df['image_id'].astype(str).isin(processed_image_ids)]
            print(f"Skipping {len(processed_image_ids)} already processed images, {len(filtered_df)} remaining")
        
        if len(filtered_df) == 0:
            print("All images already processed!")
            return results
        
        # Initialize tracking
        processed = len(processed_image_ids)
        failed = 0
        total_to_process = len(filtered_df) + processed
        
        # Calculate save frequency
        save_frequency = 1 if batch_size > 1 else 10  # Save every batch if batch_size > 1, else every 10
        batches_since_save = 0
        
        # Calculate number of batches
        num_batches = (len(filtered_df) + batch_size - 1) // batch_size
        
        print(f"Processing in {num_batches} batch(es) of size {batch_size}")
        print(f"Will save progress every {save_frequency} batch(es)")
        
        # Process in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(filtered_df))
            
            # Get batch dataframe
            batch_df = filtered_df.iloc[start_idx:end_idx]
            
            try:
                if batch_size == 1:
                    # Single row mode
                    image_id = batch_df.iloc[0]['image_id']
                    result = run_func(batch_df)
                    results[str(image_id)] = result[image_id]
                    processed += 1
                else:
                    # Batch mode
                    batch_results = run_func(batch_df)
                    
                    # Validate and add results
                    expected_ids = set(batch_df['image_id'].astype(str))
                    returned_ids = set([str(key) for key in batch_results.keys()])
                    for image_id, result in batch_results.items():
                        results[str(image_id)] = result
                    
                    processed += len(batch_results)
                    
                    # Track failures
                    missing_ids = expected_ids - returned_ids
                    if missing_ids:
                        print(f"Warning: Batch {batch_idx+1}/{num_batches} didn't return results for IDs: {missing_ids}")
                        failed += len(missing_ids)
                
                batches_since_save += 1
                
                # Save progress periodically
                if batches_since_save >= save_frequency:
                    self._save_run_progress(results, results_file, progress_file, batch_idx, num_batches)
                    batches_since_save = 0
                    print(f"Progress saved: {processed}/{total_to_process} images processed")
                
                # Progress indicator
                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    print(f"Processed {processed}/{total_to_process} images "
                        f"({batch_idx+1}/{num_batches} batches)...")
                    
            except Exception as e:
                batch_size_actual = len(batch_df)
                failed += batch_size_actual
                print(f"Error processing batch {batch_idx+1}/{num_batches} "
                    f"({batch_size_actual} images): {e}")
                continue
        
        # Final save
        self._save_run_progress(results, results_file, progress_file, num_batches-1, num_batches)
        
        # Clean up progress file since we're done
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"\nProcessing complete for run '{run_name}': {processed} successful, {failed} failed")
        print(f"Results saved to: {results_file}")
        
        return results

    def _save_run_progress(self, results: Dict, results_file: str, progress_file: str, 
                        current_batch: int, total_batches: int):
        """Save current results and progress for a run."""
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save progress info
        progress_info = {
            'last_completed_batch': current_batch,
            'total_batches': total_batches,
            'images_processed': len(results),
            'last_saved': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_info, f, indent=2)

# Example usage:
if __name__ == "__main__":
    topleft = (47.65604197918356, -122.31212182458437)
    botright = (47.65112154592941, -122.30348229827895)
    
    # Initialize crawler with project name and working directory
    crawler = SVICrawler(
        project_name="seattle_small_test",
        project_workdir="./streetview_projects",
        bounding_topleft_latlong=topleft,
        bounding_botright_latlong=botright
    )
    
    custom_filter = (
        '["highway"]'
        '["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator]'  # Must be a highway.
        'motor|no|planned|platform|proposed|raceway|razed|steps"]'
        '["area"!~"yes"]'  # Must not be an area.
        '["access"!~"private"]'  # Must not be private access.
        '["bicycle"!~"no"]'  # The "bicycle" tag must not be "no".
        '["service"!~"private"]'  # Must not be a private service road.
    )

    # This will now load existing points if they exist, or generate new ones if they don't
    # df = crawler.generate_sample_points_across_area(interval_meters=20, custom_filter=custom_filter)

    latlongs = [(47.661111490868265, -122.31387282541915), (40.74469287548298, -73.98498346574681), (42.3581746207175, -83.04530379105017)]
    df2 = crawler.generate_sample_points_from_latlongs(latlongs, custom_filter=custom_filter, force_regenerate=True, verbose=True)
    crawler.save_dataframe('sample_points', format='parquet')
    crawler.save_dataframe('sample_points', format='csv')
    exit()

    # Setup Mapillary
    import mapillary
    import mapillary.interface as mly
    load_dotenv()
    MAPILLARY_TOKEN = os.getenv('MAPILLARY_TOKEN')
    mapillary.utils.auth.set_token(MAPILLARY_TOKEN)
    mapillary_crawler = MapillaryStreetView('mapillary', MAPILLARY_TOKEN, verbose=True)
    
    def filter_download_box(topleft, botright) -> Callable[[pd.Series], bool]:
        def is_point_in_bbox(row: pd.Series) -> bool:
            max_lat, min_lon = topleft[0], topleft[1]  # topleft has max lat, min lon
            min_lat, max_lon = botright[0], botright[1]  # botright has min lat, max lon
            return (min_lat <= row['sample_lat'] <= max_lat and min_lon <= row['sample_lon'] <= max_lon)
        return is_point_in_bbox

    crawler.download_images_within_region(mapillary_crawler, 
                                          sample_point_filter_func=filter_download_box((47.653714007211626, -122.30770028310445), (47.653450946930455, -122.30703556443177)), 
                                          single_pano_per_point=True)
    
    def print_row(row):
        print(row)
        return {'output': row.get('heading').item()}
    
    crawler.run_func_on_svis('mapillary', 'print_row', print_row, resume=False)
    
    # Now you can save/load image metadata just like sample points
    # crawler.save_dataframe('image_metadata', format='parquet')
    # crawler.save_dataframe('associations', format='csv')