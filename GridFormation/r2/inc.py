import numpy as np
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
from shapely.affinity import rotate


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

def get_central_line_vertical(row):
    tol = 0.5
    bounds = row.geometry.minimum_rotated_rectangle.bounds
    min_x, min_y, max_x, max_y = bounds
    central_x = (min_x + max_x) / 2
    central_y = (min_y + max_y) / 2
    return LineString([(central_x, min_y-tol), (central_x, max_y+tol)])

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


# Rotate geometries back and extract the two bottommost points in y (after rotation)

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