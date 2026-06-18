import pandas as pd
import numpy as np
import geopandas as gpd


from shapely import wkt
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
import warnings
from shapely.ops import polygonize, unary_union
from copy import deepcopy

@pd.api.extensions.register_dataframe_accessor("nop")
class NOPAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self. tol_deg =5
        self.street_length = 10
        self.sampling_distance = 10
        self.dx = 1            # Thickness of each vertical bar
        self.angles = np.arange(0, 190, 10)
        self.survey_gdf = None
        self.survey_union_gdf = None
        self.survey_union_offset_gdf = None
        self.survey_boundary_gdf = None
        self.output_gdf = gpd.GeoDataFrame()
        self.cell_gdf = None
        self.count_gdf = None

        self.center = None
    def prepare_survey_gdf(self):
        self.survey_gdf = deepcopy(self._obj)
        # Compute the union of the survey geometries and simplify the result
        survey_union = self.survey_gdf.unary_union.simplify(2)  # Adjust tolerance 1.0 as needed

        # Create a GeoDataFrame from the union
        self.survey_union_gdf = gpd.GeoDataFrame(
            {'geometry': [survey_union]},
            crs=self.survey_gdf.crs
        )

        # Make an offset of 10 (buffer by 10 units)
        self.survey_union_offset_gdf = gpd.GeoDataFrame(
            {'geometry': [survey_union.buffer(self.street_length/2, cap_style=2)]},
            crs=self.survey_gdf.crs,
            geometry='geometry'
        )

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

# Generate the grid
    def generate_grid(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for angle in self.angles:
                    #print(f'Processing angle: {angle}')
                    # Parameters for evenly spaced thin vertical rectangles (grid "bars")
                    bounds = self.survey_union_offset_gdf.total_bounds
                    min_x, min_y, max_x, max_y = map(float, bounds)  # Ensure cast to float
                    r_x = np.sqrt(max_x**2 + max_y**2)
                    r_y = np.sqrt(min_x**2 + min_y**2)

                    # Compute number of columns safely and cast to int
                    num_cols = int(np.floor((2*r_x) / self.sampling_distance)) + 1

                    # Compute bar center x-positions
                    x_centers = np.linspace(-r_x, r_x, num=num_cols)

                    # For each center, form a thin rectangle
                    vertical_bars = [
                        Polygon([
                            (x - self.dx/2, min_y),
                            (x + self.dx/2, min_y),
                            (x + self.dx/2, max_y),
                            (x - self.dx/2, max_y)
                        ])
                        for x in x_centers
                    ]

                # Define the vector
                    vec = np.array([1,0])
                    ortho_vec = np.array([0,1])
                    # Define a rotation angle in radians (example: 45 degrees)
                    theta = np.deg2rad(angle)
                    rotation_matrix = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ])

                    # Rotate the vector by the rotation matrix
                    rotated_vec = rotation_matrix @ vec
                    ortho_vec = rotation_matrix @ ortho_vec

                    # Rotate the grid lines accordingly
                    vertical_bars_r = [rotate(bar, angle=angle, origin=(0,0)) for bar in vertical_bars]
                    grid_lines_gdf_r = gpd.GeoDataFrame({'geometry': vertical_bars_r}, crs=self.survey_gdf.crs, geometry='geometry')
                    grid_lines_gdf_r['grid_idx'] = grid_lines_gdf_r.index

                    #Create the intersection of the grid with the survey offset
                    intersection_gdf = gpd.overlay(
                        grid_lines_gdf_r, 
                        self.survey_union_offset_gdf, 
                        how='intersection', 
                        keep_geom_type=False
                    )
                    intersection_gdf = intersection_gdf.explode(index_parts=True)
                    intersection_gdf.reset_index(drop=True, inplace=True)
                    intersection_gdf.set_geometry('geometry', inplace=True)

                    try:
                        #Get the bottom points of the intersected rectamgles
                        intersection_gdf['bottom_points'] = intersection_gdf['geometry'].apply(get_bottom_points)
                    except Exception as e:
                        print(f'No bottom points for angle {angle}')
                        print(intersection_gdf['geometry'])
                        print(e)

                    # Get the unit vector along the [1,0] direction or the rotated version
                    intersection_gdf['unit_vector'] = intersection_gdf['bottom_points'].apply(safe_unit_vector)

                    # Get teh angle between the unit vector and the rotated vector
                    intersection_gdf['angle'] = intersection_gdf['unit_vector'].apply(lambda vec: angle_between_vectors(vec, rotated_vec=rotated_vec))

                    # Get only those lines which are aligned with the rotated vector
                    right_angles = intersection_gdf[np.isclose(intersection_gdf['angle'], 0, atol=self.tol_deg)]
                    if len(right_angles) > 0:
                        #print('There is a right angle')
                        #Get the cell boundaries
                        right_angles['cell_boundary'] = right_angles.apply(lambda row: get_line_from_angle(row, ortho_vec), axis=1)

                        #Rename the grid_idx to include the angle
                        right_angles['grid_idx'] = right_angles.apply(lambda row: f"{row['grid_idx']}_{angle}", axis=1)
                        right_angles.reset_index(drop=True)
                        self.output_gdf = pd.concat([self.output_gdf, right_angles])

        # Remove entries whose cell_boundary length > 1 std above the mean
        lengths = self.output_gdf.cell_boundary.length
        mean_length = lengths.mean()
        std_length = lengths.std()
        filtered_output_gdf = self.output_gdf[lengths <= (mean_length + std_length/2)]


        #Remove all the intersecting 
        #Create the cells
        geom = self.survey_union_offset_gdf.iloc[0].geometry

        # All grid lines as one noded multiline (tweak attribute if your geometry column differs)
        lines = [getattr(row, "cell_boundary") for row in filtered_output_gdf.itertuples()]
        splitters = unary_union(lines)
        network = unary_union([geom.boundary, splitters])
        cell_polys = [
            poly
            for poly in polygonize(network)
            if geom.contains(poly.representative_point())
        ]
        print(len(lines), "splitters ->", len(cell_polys), "cells")
        self.cell_gdf = gpd.GeoDataFrame({'geometry': cell_polys}, crs=self.survey_union_offset_gdf.crs)
        self.cell_gdf = self.cell_gdf.reset_index().rename(columns={'index': 'cell_idx'})
        self.cell_gdf.set_geometry('geometry', inplace=True)

        translated_cell_gdf = deepcopy(self.cell_gdf)
        translated_cell_gdf['geometry'] = translated_cell_gdf['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
        translated_cell_gdf.set_geometry('geometry', inplace=True)
        return translated_cell_gdf
   


    def count_nop(self):
        # For each cell, create an inner offset to prevent bad counts
        cell_offset_gdf = gpd.GeoDataFrame(geometry=self.cell_gdf.buffer(-0.1).explode(index_parts=False), crs=self.cell_gdf.crs)
        cell_boundaries_gdf = gpd.GeoDataFrame(geometry=cell_offset_gdf.boundary.explode(), crs=cell_offset_gdf.crs)

        #For each cell, partition the polygon into segments along its boundary lines using exterior coordinates. This will help us to get the nummber of boundaries
        segment_list = []
        for poly in cell_offset_gdf.geometry:
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                seg = LineString([coords[i], coords[i+1]])
                segment_list.append(seg)

        partitioned_gdf = gpd.GeoDataFrame(geometry=segment_list, crs=cell_offset_gdf.crs)

        #Get the intersections of the surveys with the boundaries
        points = gpd.GeoDataFrame(
            gpd.overlay(self.survey_gdf, cell_boundaries_gdf, how='intersection', keep_geom_type=False).explode(),
            crs=self.survey_gdf.crs
        )
        points = points.set_geometry('geometry')
        points_buffer = gpd.GeoDataFrame(geometry=points.buffer(0.05), crs=points.crs)

        #Get the boundaries that intersect with the survey
        points_buffer.reset_index(drop=True, inplace=True)
        boundaries = gpd.sjoin(points_buffer, partitioned_gdf, how='right', predicate='intersects')

         #Get the boundaries that intersect with the survey
        points_buffer.reset_index(drop=True, inplace=True)
        boundaries = gpd.sjoin(points_buffer, partitioned_gdf, how='right', predicate='intersects')
        # Fix: Only cast to int if value is finite (not NA/inf) to avoid IntCastingNaNError
        if boundaries['index_left'].notnull().all():
            boundaries['index_left'] = boundaries['index_left'].astype(int)
        else:
            # Fill NA with a placeholder (e.g., -1) before casting, or just keep NA if that's acceptable
            boundaries['index_left'] = boundaries['index_left'].fillna(-1).astype(int)
            # Drop rows where index_left == -1
        boundaries = boundaries[boundaries['index_left'] != -1]
        boundaries.rename(columns={'index_left': 'points_idx'}, inplace=True)
        # Get unique geometries from 'boundaries' and put in a GeoDataFrame
        unique_geoms = boundaries['geometry'].unique()
        unique_gdf = gpd.GeoDataFrame(geometry=list(unique_geoms), crs=boundaries.crs)
        unique_gdf.reset_index(drop=True, inplace=True)
        joined_gdf_unique = gpd.sjoin(unique_gdf, self.cell_gdf, how='right', predicate='intersects')
        boundary_counts = joined_gdf_unique.groupby('cell_idx').size().reset_index(name='boundaries')

        # Get the number of intersections per cell
        joined_gdf = gpd.sjoin(points, self.cell_gdf, how='left', predicate='intersects')
        # Get intersection counts by cell
        intersection_counts = joined_gdf.groupby('cell_idx').size().reset_index(name='intersections')
        intersection_boundaties = intersection_counts.merge(boundary_counts, left_on='cell_idx', right_on='cell_idx', how='left')

        #Get the output
        cell_summary = self.cell_gdf.merge(intersection_boundaties, left_on='cell_idx', right_on='cell_idx', how='left')
        cell_summary['passes'] = cell_summary['intersections'] / cell_summary['boundaries']
        
        translated_count_gdf = deepcopy(cell_summary)
        translated_count_gdf['geometry'] = translated_count_gdf['geometry'].translate(xoff=self.center.x, yoff=self.center.y)
        translated_count_gdf.set_geometry('geometry', inplace=True)
        return translated_count_gdf

    def get_aggregated_nop(self):
        # Create a new GeoDataFrame with necessary columns
        cell_info_nonan = self.count_gdf.dropna(subset=['passes']).copy()

        dissolved = self.count_gdf.dissolve(by='passes')

            # Ensure the dissolved result is a GeoDataFrame indexed by 'passes'
        aggregated_cells_by_pass_gdf = dissolved.reset_index()[['passes', 'geometry']].translate(xoff=self.center.x, yoff=self.center.y)
        return aggregated_cells_by_pass_gdf


#--- Defition of functions used for the accessor ---#


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