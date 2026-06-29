import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx

from shapely import wkt
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString
from shapely.affinity import rotate
import warnings
from shapely.ops import polygonize, unary_union
from copy import deepcopy
from shapely.strtree import STRtree


@pd.api.extensions.register_dataframe_accessor("nop")
class NOPAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        #--Configuration parameters--#
    
        self.street_length =7
        self.sampling_distance = 10
        self.cell_buffer_distance = 2
        self.dx = 0.5            # Thickness of each vertical bar

        #--Angle spacing--#
        self.angles = -10 * np.ones(len(np.arange(0, 181, 10)))
        self. tol_deg =5

        self.survey_gdf = None
        self.survey_union_gdf = None
        self.survey_union_offset_gdf = None
        self.survey_boundary_gdf = None
        self.output_gdf = gpd.GeoDataFrame()
        self.cell_gdf = None
        self.count_gdf = None
        self.utm_crs = None
        self.net = None

        self.center = None
        self.reference_gdf = None

    def prepare_survey_gdf(self):
        self.survey_gdf = deepcopy(self._obj)
        # Compute the union of the survey geometries and simplify the result
        survey_union = self.survey_gdf.unary_union

        # Create a GeoDataFrame from the union
        self.survey_union_gdf = gpd.GeoDataFrame(
            {'geometry': [survey_union]},
            crs=self.survey_gdf.crs
        )

        # Make an offset of 10 (buffer by 10 units)
        self.survey_union_offset_gdf = gpd.GeoDataFrame(
            {'geometry': [survey_union.buffer(self.street_length/2, join_style=2)]},
            crs=self.survey_gdf.crs,
            geometry='geometry'
        )
        self.utm_crs = self.survey_gdf.crs
        # Get the center as the midpoint of the bounds (average of min/max x/y)
        minx, miny, maxx, maxy = self.survey_union_offset_gdf.total_bounds
        self.center = Point((minx + maxx) / 2, (miny + maxy) / 2)
        print("Center of the bounds:", self.center)

        # Shift the survey_union_offset_gdf geometry so that the centroid is at (0, 0)
        self.survey_union_offset_gdf['geometry'] = self.survey_union_offset_gdf.geometry.translate(
            xoff=-self.center.x, yoff=-self.center.y
        )

        self.survey_gdf['geometry'] = self.survey_gdf.geometry.translate(
        xoff=-self.center.x, yoff=-self.center.y
        )
        self.survey_gdf.set_geometry('geometry', inplace=True)
        self.survey_boundary_gdf = gpd.GeoDataFrame({'geometry': [self.survey_union_offset_gdf.geometry.boundary.values[0]]}, crs=self.survey_gdf.crs)

    def _generate_grid_lines(self):
        bounds = self.survey_union_offset_gdf.total_bounds
        min_x, min_y, max_x, max_y = map(float, bounds)  # Ensure cast to float
        r_x = np.sqrt(max_x**2 + max_y**2)
        r_y = np.sqrt(min_x**2 + min_y**2)

        # Compute number of columns safely and cast to int
        num_cols = int(np.floor((2*r_x) / self.sampling_distance)) + 1

        # Compute bar center x-positions
        x_centers = np.linspace(-r_x, r_x, num=num_cols)

        # For each center, form a thin rectangle
        grid_lines = [
        Polygon([
            (x - self.dx/2, r_y),
            (x + self.dx/2, r_y),
            (x + self.dx/2, -r_x),
            (x - self.dx/2, -r_x)
        ])
        for x in x_centers
        ]
        # Define the vector
        grid_lines_gdf = gpd.GeoDataFrame({'geometry': grid_lines}, crs=self.survey_gdf.crs, geometry='geometry')
        grid_lines_gdf['grid_idx'] = grid_lines_gdf.index
        return grid_lines_gdf



  
# Generate the grid
    def generate_grid(self):

        output_gdf = gpd.GeoDataFrame()

        #----------------Generate the vertical grid lines----------------#
        angle = 0
        vec = np.array([1,0])
        ortho_vec = np.array([0,1])
        grid_lines_gdf = self._generate_grid_lines()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #--------------- Generate the cell boundaries -------------------#
            for d_angle in self.angles:
                print('Processing angle: ', angle)
                # Rotate the map
                if angle == 0:
                    contour_gdf = self.survey_union_offset_gdf.copy()
                    survey_rotated_gdf = self.survey_gdf.copy()
                else:
                    contour_gdf['geometry'] = contour_gdf['geometry'].rotate(angle=-d_angle, origin=(0,0))
                    survey_rotated_gdf['geometry'] = survey_rotated_gdf['geometry'].rotate(angle=-d_angle, origin=(0,0))

                #Create the intersection of the grid with the survey offset
                intersection_gdf = gpd.overlay(
                    grid_lines_gdf, 
                    contour_gdf, 
                    how='intersection', 
                    keep_geom_type=False
                )    
                
                intersection_gdf = intersection_gdf.explode(index_parts=True)
                intersection_gdf.reset_index(drop=True, inplace=True)
                intersection_gdf.set_geometry('geometry', inplace=True)
                intersection_gdf.to_crs(self.survey_gdf.crs, inplace=True)

                #Measure the angles
                intersection_gdf['bottom_points'] = intersection_gdf['geometry'].apply(lambda geom: get_bottom_two_points(geom,angle,rotattion = False))

                # Get the unit vector along the [1,0] direction or the rotated version
                intersection_gdf['unit_vector'] = intersection_gdf['bottom_points'].apply(safe_unit_vector)

                # Get teh angle between the unit vector and the rotated vector
                intersection_gdf['angle_measured'] = intersection_gdf['unit_vector'].apply(lambda vec: angle_between_vectors(vec,rotated_vec=[1,0]))
                intersection_gdf['angle_rotated'] = angle
    
                # Get only those lines which are aligned with the rotated vector
                right_angles = intersection_gdf[np.isclose(intersection_gdf['angle_measured'], 0, atol=self.tol_deg)]
                if len(right_angles) > 0:
                    right_angles['cell_boundary'] = right_angles.apply(lambda row: get_central_line_vertical(row), axis=1)
                    right_angles['cell_boundary_length'] = right_angles['cell_boundary'].apply(lambda line: line.length) - 1
                    #mean_cell_boundary_length = right_angles['cell_boundary_length'].mean()
                    #std_cell_boundary_length = right_angles['cell_boundary_length'].std()
                    #right_angles = right_angles[np.isclose(right_angles['cell_boundary_length'], 12.7, atol=2*.9)]
                    right_angles['grid_boundary'] = right_angles['geometry']
                    right_angles['geometry'] = right_angles['cell_boundary']
                    right_angles = right_angles.set_geometry('geometry')
                    right_angles = right_angles.set_crs(self.utm_crs)    
                    right_angles['cell_idx'] = right_angles.index

                    #Intersection of the boundaries with the surveys
                    intersection_survey_gdf = gpd.overlay(right_angles, survey_rotated_gdf[['geometry']], how='intersection', keep_geom_type=False)
                    intersection_survey_gdf = intersection_survey_gdf.explode(index_parts=True)
                    intersection_survey_gdf = intersection_survey_gdf[['geometry']]

                    intersection_survey_gdf = intersection_survey_gdf.reset_index(drop=True)
                    intersection_survey_gdf = intersection_survey_gdf.set_geometry('geometry')
                    intersection_survey_gdf = intersection_survey_gdf.set_crs(self.utm_crs)


                    #Join the boundaries with the surveys intersection
                    r = right_angles.copy()
                    r['geometry'] = r['cell_boundary']
                    r.drop(columns = ['cell_boundary','grid_boundary'], inplace=True)
                    r = r.set_geometry('geometry')
                    r = r.set_crs(self.utm_crs)
                    joined = gpd.sjoin(r, intersection_survey_gdf, how='right', predicate='intersects')

                    spread = joined.groupby('cell_idx').apply(lambda x: get_max_min_points(x))
                    nop = joined.groupby('cell_idx').apply(lambda x: get_nop(x))
                    spread_df = spread.reset_index()
                    spread_df.rename(columns = {0:'spread'}, inplace=True)
                    nop_df = nop.reset_index()
                    nop_df.rename(columns = {0:'nop'}, inplace=True)
                    result = pd.merge(spread_df, nop_df, on = 'cell_idx', how = 'left')
                    r = pd.merge(r, result, on = 'cell_idx', how = 'left')
        
                    #filtered = right_angles[(right_angles['cell_boundary_length'] < 2*street_length)]
                    output_gdf = pd.concat([output_gdf, r])
                
                    #output_gdf = output_gdf[(output_gdf['spread'] < street_length + 3*std_spread) & (output_gdf['cell_boundary_length'] < street_length + 3*std_len)]
                    angle = angle + d_angle
            output_gdf.drop(columns = ['cell_idx'], inplace = True)

        #--------------- Remove the cell boundaries that are too small or too large -------------------#
        # Efficiently filter points with fewer intermediate variables and reduce DataFrame copies
        # Compute 'diff' column in-place
        output_gdf['diff'] = (output_gdf['cell_boundary_length'] - output_gdf['spread']) / self.street_length

        # Apply all filters in a single step using boolean indexing for efficiency
        mask = (
            ~output_gdf['nop'].isna() &
            # (output_gdf['cell_boundary_length'] < 2*street_length) &  # Uncomment if needed
            (output_gdf['diff'] >= 1) &
            (output_gdf['diff'] <= 1.05)
        )

        output_gdf_long = output_gdf[mask]

        # Print summary efficiently
        print(f"Total boundaries: {output_gdf.shape[0]}")
        print(f"Boundaries with no NaN 'nop': {(~output_gdf['nop'].isna()).sum()}")
        print(f"Boundaries after length and spread filters: {output_gdf_long.shape[0]}")

        #--------------- Rotate back the cells to the original orientation -------------------#
        # Rotate all geometries "in one go" using vectorized .apply on the DataFrame
        # Note: negative sign, since original was rotate to -angle_rotated
        rotated_output_gdf = output_gdf_long.copy()
        rotated_output_gdf['geometry'] = [
            rotate(geom, angle, origin=(0, 0))
            for geom, angle in zip(rotated_output_gdf['geometry'].values, rotated_output_gdf['angle_rotated'].values)]

        #--------------- Remove the intersecting boundaries -------------------#
        #Remove all the overlaps by taking:
        #1. The cell with the most intersections
        #2. The cell with the largest boundary length if there is a tie
        overlapped = gpd.sjoin(rotated_output_gdf, rotated_output_gdf, how = 'inner', predicate = 'intersects')
        overlapped = overlapped[overlapped.index != overlapped.index_right].copy()
        overlapped = overlapped.reset_index()

        overlapped = overlapped[['index','index_right','cell_boundary_length_left']]
        indx_to_remove = []

        while (len(overlapped) > 0):
            overlapped_summary = overlapped.groupby('index').agg({'cell_boundary_length_left':'mean','index_right':'nunique'}).sort_values(by = 'index_right', ascending = False)
            idx_intersection = overlapped_summary['index_right'].idxmax()
            max_intersections = overlapped_summary['index_right'].max()
            if max_intersections == 1:
                idx_remove = overlapped_summary['cell_boundary_length_left'].idxmax()
                max_remove = overlapped_summary['cell_boundary_length_left'].max()
            else:
                idx_remove = idx_intersection
                max_remove = max_intersections
            overlapped = overlapped[overlapped['index'] != idx_remove]
            overlapped = overlapped[overlapped['index_right'] != idx_remove]
            indx_to_remove.append(idx_remove)
        rotated_wnoverlap = rotated_output_gdf[~(rotated_output_gdf.index.isin(indx_to_remove))]
        print(f"Boundaries after removing overlaps: {len(rotated_wnoverlap)}")

        #--------------- Rempove the cells thar are close together -------------------#
        # Add a column to keep the original index and centroid
        removed_gdf_centroids = rotated_wnoverlap.copy()
        removed_gdf_centroids['centroid'] = removed_gdf_centroids.geometry.centroid
        removed_gdf_centroids['orig_idx'] = removed_gdf_centroids.index

        # Get list of centroid geometries and corresponding original indices
        centroids_geom = removed_gdf_centroids['centroid'].tolist()
        orig_indices = removed_gdf_centroids['orig_idx'].tolist()

        # Build a spatial index for fast neighbor finding
        str_tree = STRtree(centroids_geom)

        # Use a numpy array to efficiently build a mask of items to keep
        remove_mask = np.zeros(len(centroids_geom), dtype=bool)

        # Pre-query all centroid buffers once and avoid repeated STRtree.query
        # Approach: sort all by index once. As soon as you mark one for remove, skip future work.
        for idx, centroid in enumerate(centroids_geom):
            if remove_mask[idx]:
                continue  # Already marked for removal
            # Query for any neighbors within 10 units (excluding itself)
            nearby_indices = [i for i in str_tree.query(centroid.buffer(self.cell_buffer_distance), predicate=None)]
            for nidx in nearby_indices:
                if nidx != idx and not remove_mask[nidx]:
                    remove_mask[nidx] = True

        # Now drop rows by the mask
        filtered_removed_gdf = removed_gdf_centroids[~remove_mask]
        removed_gdf = filtered_removed_gdf.reset_index(drop=True)
        print(f"Boundaries after removing close together cells: {len(removed_gdf)}")

        #--------------- Create the cells -------------------#
        geom = self.survey_union_offset_gdf.iloc[0].geometry

        # Direct extraction of geometry numpy array is typically faster than itertuples or apply
        lines = removed_gdf.geometry.values
        splitters = unary_union(lines)

        # Only unary_union geom.boundary once and directly pass as list with splitters
        network = unary_union([geom.boundary, splitters])

        # Polygonize all at once, and filter efficiently using vectorized contains (if available)
        cell_polys_gen = list(polygonize(network))
        # Use shapely's prepared geometries for faster contains checks
        from shapely.prepared import prep
        prep_geom = prep(geom)
        cell_polys = [poly for poly in cell_polys_gen if prep_geom.contains(poly.representative_point())]

        print(f"{len(lines)} splitters -> {len(cell_polys)} cells")
        self.cell_gdf = gpd.GeoDataFrame({'geometry': cell_polys}, crs=self.survey_union_offset_gdf.crs)
        self.cell_gdf = self.cell_gdf.reset_index().rename(columns={'index': 'cell_idx'})
        self.cell_gdf.set_geometry('geometry', inplace=True)

        self.output_gdf = removed_gdf

        translated_cell_gdf = deepcopy(self.cell_gdf)
        translated_cell_gdf['geometry'] = translated_cell_gdf['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
        translated_cell_gdf.set_geometry('geometry', inplace=True)
        return translated_cell_gdf

    def get_boundary_gdf(self):
        translated_cell_gdf = deepcopy(self.output_gdf)
        translated_cell_gdf['geometry'] = translated_cell_gdf['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
        translated_cell_gdf['centroid'] = translated_cell_gdf['geometry'].centroid
        translated_cell_gdf.set_geometry('geometry', inplace=True)
        return translated_cell_gdf


    def intersect_survey_with_cells(self, cell_gdf=None):
        """
        Clip each survey breadcrumb to each cell individually.

        Unlike ``gpd.overlay`` on the full survey, this does not split lines at
        crossings between breadcrumbs, so pass counts stay correct at intersections.
        """
        cells = cell_gdf if cell_gdf is not None else self.cell_gdf
        order_col = "Order" if "Order" in self.survey_gdf.columns else None
        records = []
        for _, cell_row in cells.iterrows():
            cell_idx = cell_row["cell_idx"]
            cell_geom = cell_row.geometry
            for _, seg_row in self.survey_gdf.iterrows():
                inter = seg_row.geometry.intersection(cell_geom)
                if inter.is_empty:
                    continue
                base = {"cell_idx": cell_idx}
                if order_col is not None:
                    base[order_col] = seg_row[order_col]
                for line in iter_linestrings(inter):
                    records.append({**base, "geometry": line})
        if not records:
            return gpd.GeoDataFrame(
                columns=["cell_idx", "geometry"],
                crs=self.survey_gdf.crs,
                geometry="geometry",
            )
        return gpd.GeoDataFrame(records, crs=self.survey_gdf.crs, geometry="geometry")

    def count_nop(self,type='cell', argument=None):
        if type == 'cell':
            cell_buffer = deepcopy(self.cell_gdf.copy())
            cell_buffer['geometry'] = cell_buffer['geometry'].buffer(-0.1)
            cell_buffer['geometry'] = cell_buffer['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
            cell_buffer.set_geometry('geometry', inplace=True)
            cell_buffer.crs = self.cell_gdf.crs
            if argument is None:
                survey_cell = gpd.overlay(self.survey_gdf, cell_buffer, how='intersection')
            elif argument is not None and 'Breadcrumb' in argument:
                bc = argument['Breadcrumb']
                bc.to_crs(self.cell_gdf.crs, inplace=True)
                survey_cell = gpd.overlay(argument['Breadcrumb'][['geometry','Order']], cell_buffer, how='intersection')
                nop = pd.DataFrame({'n_passes': survey_cell.groupby('cell_idx').apply(count_lines)})
            cell_wnop = pd.merge(self.cell_gdf.copy(), nop, on='cell_idx', how='left')
            translated_cell_wnop = cell_wnop.copy()
            translated_cell_wnop['geometry'] = translated_cell_wnop['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
            translated_cell_wnop.set_geometry('geometry', inplace=True)
            return cell_buffer,survey_cell,translated_cell_wnop

    def create_network(self):
        #Create the nodes from the cell centroid
        nodes = self.cell_gdf.copy()
        nodes['centroid'] = nodes.geometry.centroid
        nodes.set_geometry('centroid', inplace = True)
        nodes.crs = self.cell_gdf.crs

        #Create the network by checking rook contiguity
        cell_buffer = self.cell_gdf.copy()
        cell_buffer['geometry'] = cell_buffer['geometry'].buffer(0.1, cap_style=2)
        edges_gdf = gpd.sjoin(cell_buffer, cell_buffer, how = 'left', predicate = 'intersects')
        edges_gdf.rename(columns = {'index_right':'v'}, inplace = True)
        edges_gdf['u'] = edges_gdf.index
        edges_gdf = edges_gdf[edges_gdf['u'] < edges_gdf['v']][['geometry', 'u', 'v']]
        edges_wnop_gdf = gpd.sjoin(edges_gdf, self.output_gdf, how = 'left', predicate = 'intersects')
        edge_nop = edges_wnop_gdf.groupby(["u", "v"], as_index=False)["nop"].max()

        # Create a networkx graph
        net = nx.Graph()
        # Add nodes for each cell (node ids from 'u' and 'v' columns)
        for idx,row in nodes.iterrows():
            net.add_node(idx)
            net.nodes[idx]['centroid'] = row['centroid']

        # Add edges between neighboring cells
        for _, row in edge_nop.iterrows():
            u = row['u']
            v = row['v']
            net.add_edge(u, v)
            net[u][v]['numberOfIntersection'] = row['nop']
        # Optionally display basic info about the network
        print(f"Created network with {net.number_of_nodes()} nodes and {net.number_of_edges()} edges.")
        self.net = net
        return net

    def get_aggregated_nop(self):
        # Create a new GeoDataFrame with necessary columns
        cell_info_nonan = self.count_gdf.dropna(subset=['passes']).copy()

        dissolved = self.count_gdf.dissolve(by='passes')

            # Ensure the dissolved result is a GeoDataFrame indexed by 'passes'
        aggregated_cells_by_pass_gdf = dissolved.reset_index()[['passes', 'geometry']].translate(xoff=self.center.x, yoff=self.center.y)
        return aggregated_cells_by_pass_gdf

    def get_unique_survey_line(self):
        # Efficiently concatenate the LineStrings based on 'Order'
        ordered_geoms = self.survey_gdf.sort_values('Order')['geometry'].values
        all_coords = []
        for geom in ordered_geoms:
            if not all_coords:
                all_coords.extend(geom.coords)
            else:
                # drop the first point if it's a duplicate of previous segment's last point
                coords = list(geom.coords)
                if coords[0] == all_coords[-1]:
                    coords = coords[1:]
                all_coords.extend(coords)
        total_linestring = LineString(all_coords)
        total_linestring_gdf = gpd.GeoDataFrame({'geometry': [total_linestring]}, crs=self.survey_gdf.crs)  
        return total_linestring_gdf

#--- Defition of functions used for the accessor ---#
def count_lines(df):
    bc = df['Order'].nunique()
    if bc > 1:
        # Subtract 1 from the total for each occurrence of consecutive order numbers
        order_list = sorted(df['Order'].unique())
        num_consecutive = sum((order_list[i+1] - order_list[i] == 1) for i in range(len(order_list)-1))
        return bc - num_consecutive
    else:
        # Check if 'geometry' is MultiLineString and count breadcrumbs accordingly
        geom = df['geometry'].iloc[0]
        if geom.geom_type == 'MultiLineString':
            # Treat each LineString as a breadcrumb
            return len(geom.geoms)
        else:
            return bc
  

def is_aligned(row, rotated_vec, atol=5):
    """
    True if any side of the geometry's **minimum rotated rectangle** is parallel to
    ``rotated_vec`` (within ``atol`` degrees).

    Uses ``minimum_rotated_rectangle.exterior`` so we never call ``.boundary.coords`` on a
    multipart boundary (e.g. MultiPolygon or polygon with holes), which raises
    "Multi-part geometries do not provide a coordinate sequence".
    """
    geom = getattr(row, "geometry", None)
    if geom is None or geom.is_empty:
        return False
    try:
        rect = geom.minimum_rotated_rectangle
    except Exception:
        return False
    if rect.is_empty or rect.geom_type != "Polygon":
        return False
    coords = np.asarray(rect.exterior.coords, dtype=float)
    if coords.shape[0] < 2:
        return False
    rv = np.asarray(rotated_vec, dtype=float)
    nrm = np.linalg.norm(rv)
    if nrm == 0:
        return False
    rv = rv / nrm
    # Closed ring: last edge may be zero-length duplicate of first
    for i in range(coords.shape[0] - 1):
        vec = coords[i + 1, :2] - coords[i, :2]
        ln = np.linalg.norm(vec)
        if ln == 0:
            continue
        vec = vec / ln
        angle = angle_between_vectors(vec.tolist(), rv.tolist())
        if np.isclose(angle, 0.0, atol=atol) or np.isclose(abs(angle), 180.0, atol=atol):
            return True
    return False

def vector_to_point(start_point, vector, length=1.0):
    """
    Returns the Point at the end of the vector of a given length starting from start_point.

    Parameters:
    - start_point: shapely.geometry.Point, the starting point.
    - vector: array-like or list [x, y], the direction as a vector.
    - length: float, the length to scale the vector (default 1.0, for unit vector).

    Returns:
    - shapely.geometry.Point at the tip of the (scaled) vector starting from start_point.
    """
    v = np.array(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0 or start_point is None:
        return None
    v = v / norm * length
    return Point(start_point.x + v[0], start_point.y + v[1])

# Take the unit vector from p1 (Point) to p2 (Point)
def unit_vector_between_points(p1, p2):
    try:
        if p1 is None or p2 is None:
            return np.array([np.nan, np.nan])
        x0, y0 = p1.x, p1.y
        x1, y1 = p2.x, p2.y
        dx = x1 - x0
        dy = y1 - y0
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            return np.array([np.nan, np.nan])
        return np.array([dx/norm, dy/norm])
    except Exception:
        return np.array([np.nan, np.nan])

# Compute angle in degrees between two vectors (default: unit_vector and [1,0]), range [-180, 180]
def angle_between_vectors(vec_a, rotated_vec=[1.0, 0.0]):
    # vec_a and rotated_vec expected to be [x, y]
    a = np.array(vec_a)
    b = np.array(rotated_vec)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    # Calculate angle in radians
    dot = np.dot(a, b)
    det = a[0] * b[1] - a[1] * b[0]
    angle_rad = np.arctan2(det, dot)
    return np.degrees(angle_rad)

def get_line_from_angle(row, ortho_vec):
    tol = 10
    top = tuple(np.array(row.geometry.centroid.coords[0]) + (row.geometry.length/2 + tol) * np.array(ortho_vec))
    bottom = tuple(np.array(row.geometry.centroid.coords[0]) - (row.geometry.length/2 + tol) * np.array(ortho_vec))
    return LineString([top, bottom])

def safe_unit_vector(bottom_points, tol=1e-8):
    """
    Returns a unit vector for the line defined by bottom_points. 
    If invalid input or the two points are extremely close, returns None.
    They are not really coincident points—just extremely close numerically.
    """
    # Validate: must be a list of 2 coordinate tuples
    if (
        isinstance(bottom_points, list)
        and len(bottom_points) == 2
        and all(isinstance(pt, tuple) and len(pt) == 2 for pt in bottom_points)
    ):
        p1 = np.array(bottom_points[0], dtype=float)
        p2 = np.array(bottom_points[1], dtype=float)
        vec = p2 - p1
        norm = np.linalg.norm(vec)

        # Instead of coincident check, check that points are not "almost" coincident
        if norm <= tol or np.any(np.isnan(vec)):
            return None

        return (vec / norm).tolist()
    # If we get here, input was not valid, so return to previous logic: None
    # (i.e., do what the old safe_unit_vector used to do on fallback)
    return None
    # Not enough points or bad input: return None
    return None

def get_bottom_points(geom):
    """
    Two points along the lower / trailing part of the geometry, for unit-vector use.

    Overlay output varies by type. Polygon exteriors are used instead of ``boundary`` because
    polygons with holes expose ``boundary`` as a MultiLineString, which has no ``.coords``.
    """
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "Polygon":
        # Do not use ``geom.boundary.coords``: with interior rings, ``boundary`` is a
        # MultiLineString and has no coordinate sequence (Shapely raises).
        ext = geom.exterior
        ring_coords = list(dict.fromkeys(ext.coords))
        return ring_coords[-2:] if len(ring_coords) >= 2 else None
    if t == "MultiPolygon":
        polys = [p for p in geom.geoms if not p.is_empty]
        if not polys:
            return None
        return get_bottom_points(max(polys, key=lambda p: p.area))
    if t == "LineString":
        coords = list(geom.coords)
        return coords[-2:] if len(coords) >= 2 else None
    if t == "MultiLineString":
        lines = [ln for ln in geom.geoms if not ln.is_empty]
        if not lines:
            return None
        return get_bottom_points(max(lines, key=lambda ln: ln.length))
    if t == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if polys:
            return get_bottom_points(max(polys, key=lambda p: p.area))
        mpolys = [g for g in geom.geoms if g.geom_type == "MultiPolygon" and not g.is_empty]
        if mpolys:
            return get_bottom_points(max(mpolys, key=lambda p: p.area))
        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString") and not g.is_empty]
        if lines:
            return get_bottom_points(max(lines, key=lambda g: g.length))
    return None

def segment_line(row):
    out_segment = []
    coords = row['geometry'].coords
    # Create line segments
    segments = [(coords[i], coords[i+1]) for i in range(len(coords) - 1)]
    for segment in segments:
        out_segment.append(LineString(segment))
    return out_segment

def orthogonal_axes(row, return_axis = 'minor'):
    """
    Given a row of a pandas DataFrame containing a 'geometry' column (shapely geometry),
    returns a dictionary with the center, major/minor axis unit vectors,
    and Shapely LineStrings for major and minor axes.

    Designed to work with DataFrame.apply(..., axis=1).
    """
    rect = row['geometry'].minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)
    # rectangle has 4 edges, closed; coords has 5 points, so take the first 4 pairs
    edges = [
        (coords[i], coords[i+1])
        for i in range(4)
    ]

    # compute edge vectors and lengths
    vectors = []
    lengths = []
    for p1, p2 in edges:
        vec = np.array(p2) - np.array(p1)
        vectors.append(vec)
        lengths.append(np.linalg.norm(vec))

    # sort edges by length
    order = np.argsort(lengths)

    # minor axis = shortest edge direction
    minor_vec = vectors[order[0]]
    # major axis = longest edge direction
    major_vec = vectors[order[-1]]

    # normalize
    major_axis = major_vec / np.linalg.norm(major_vec)
    minor_axis = minor_vec / np.linalg.norm(minor_vec)

    # center of rectangle (ignore last coord, duplicate start)
    center = coords[:-1].mean(axis=0)

    # build axis lines
    scale = 5

    major_line = LineString([
        tuple(center - major_axis * scale),
        tuple(center + major_axis * scale)
    ])
    minor_line = LineString([
        tuple(center - minor_axis * scale),
        tuple(center + minor_axis * scale)
    ])

    # Return structured data suitable for apply
    #return {
    #    'center': tuple(center),
    #    'major_axis': major_axis.tolist(),
    #    'minor_axis': minor_axis.tolist(),
    #    'major_line': major_line,
    #    'minor_line': minor_line
    #}
    if return_axis == 'minor':
        return minor_axis
    else:
        return major_axis


def assign_top_point(df, grid_idx_col='grid_idx', sort_col='geometry_F0_y'):
    """
    For each group of grid_idx, assign the index of the previous (higher-y) row as 'top_point_idx'.
    The top_point_idx column holds the 'index' of the previous row in the sorted group.
    """
    df = df.copy()
    # We'll add top_point_idx by referencing the original index
    df_reset = df.reset_index()
    # Assign top_point_idx as the previous row's index in the sorted group
    def assign_shifted(sub_df):
        sub_df = sub_df.sort_values(by=sort_col, ascending=False)
        sub_df['top_point_idx'] = sub_df['index'].shift(1)
        return sub_df
    # Apply per group and update
    updated = df_reset.groupby(grid_idx_col, group_keys=False).apply(assign_shifted)
    # Align back using original index
    df['top_point_idx'] = pd.NA
    # updated['index'] is the original index, updated['top_point_idx'] is previous row index or NA
    df.loc[updated['index'], 'top_point_idx'] = updated['top_point_idx'].astype('Int64').values
    return df


def assign_top_geometry(df, geometry_col='geometry', top_idx_col='top_point_idx', new_col='top_geometry'):
    """
    Assign, for each row, the geometry corresponding to its top_point_idx (from the same DataFrame).
    The new column (default: 'top_geometry') will be pd.NA if top_point_idx is NA.
    """
    df = df.copy()
    # Build a Series for fast lookup (index: DataFrame index, value: geometry)
    geometry_lookup = df[geometry_col]
    
    # Function to fetch geometry matching top_point_idx
    def get_top_geom(idx):
        if pd.isna(idx):
            return pd.NA
        try:
            return geometry_lookup.loc[int(idx)]
        except Exception:
            return pd.NA
    
    df[new_col] = df[top_idx_col].map(get_top_geom)
    return df


def safe_get_geometry(ix, df, geometry_col='geometry'):
    """
    Safely get the geometry from DataFrame df at row index ix.
    Returns pd.NA if ix is NA, out of range, or any error occurs.
    """
    if pd.isna(ix):
        return pd.NA
    try:
        return df.loc[int(ix), geometry_col]
    except Exception:
        return pd.NA

def line_between_points(row,p1,p2,dy):
    """
    Given a row with 'geometry' (Point) and 'dx' (Point), 
    returns a LineString between them if both are valid, else pd.NA.
    """
    geom = p1
    dx = p2
    # Check if both are valid Points (using hasattr to not break execution)
    if (geom is not None 
        and dx is not None 
        and not pd.isna(geom) 
        and not pd.isna(dx)):
        # Sometimes 'dx' can be pd.NA (type <NA>), which is not a geometry
        try:
            from shapely.geometry import LineString
            return LineString([geom, dx])
        except Exception:
            return pd.NA
    return pd.NA

def iter_linestrings(geom):
    """Yield LineString parts from a Shapely geometry."""
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == "LineString":
        yield geom
    elif gt == "MultiLineString":
        yield from geom.geoms
    elif gt == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_linestrings(part)


def count_line_components(gdf, geometry_col="geometry"):
    """Count LineString / MultiLineString components in a GeoDataFrame."""
    return sum(1 for geom in gdf[geometry_col] for _ in iter_linestrings(geom))


def count_survey_passes(gdf, order_col="Order", geometry_col="geometry"):
    """
    Count distinct survey passes through a cell.

    At road intersections, overlay can split breadcrumbs into extra LineString
    fragments that still share the same ``Order``; count unique orders instead.
    """
    if order_col in gdf.columns and not gdf.empty:
        return gdf[order_col].nunique()
    return count_line_components(gdf, geometry_col)


def get_max_min_points(df):
   min_x, min_y, max_x, max_y = MultiPoint(list(df['geometry'])).bounds
   return max_y - min_y

def get_nop(df):
    if "Order" in df.columns:
        return df["Order"].nunique()
    return df.shape[0]

def get_bottom_two_points(geom,angle,rotattion = True):
    coords = list(geom.exterior.coords)
    # Remove duplicates and convert to Point objects
    coords = list(dict.fromkeys(coords))
    # Find the two points with the smallest y values (i.e., bottommost points)
    bottom_two = sorted(coords, key=lambda pt: pt[1])[:2]
    # Now sort these two points only by x ascending (left to right)
    bottom_two = sorted(bottom_two, key=lambda pt: pt[0])
    if rotattion:
        bottom_two = [rotate(Point(pt), angle=angle, origin=(0, 0)) for pt in bottom_two]
    else:
        bottom_two = [Point(pt) for pt in bottom_two]
    return [tuple(pt.coords[0]) for pt in bottom_two]

def get_central_line_vertical(row):
    tol = 0.5
    bounds = row.geometry.minimum_rotated_rectangle.bounds
    min_x, min_y, max_x, max_y = bounds
    central_x = (min_x + max_x) / 2
    central_y = (min_y + max_y) / 2
    return LineString([(central_x, min_y-tol), (central_x, max_y+tol)])

