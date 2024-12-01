import math
import numpy as np

def basic_euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def nearest_obstacle_point(x,y, xy_array):
    xy_array = np.array(xy_array)
    all_distances = [basic_euclidean((x, y), point) for point in xy_array]
    min_distance = min(all_distances)
    nearest = xy_array[np.argmin(all_distances)]
    return [nearest, min_distance]

def normalize_distances(distance_array):
    max_distance = max(distance_array)
    return np.clip((distance_array/max_distance), 0, 1)

def parse_coordinates(coord_str):
    if isinstance(coord_str, tuple):  # Already
        # a tuple, no need to parse
        return coord_str
    elif isinstance(coord_str, list):  # Convert list to tuple
        return tuple(map(float, coord_str))
    elif isinstance(coord_str, str):  # Parse string representation of a tuple
        return tuple(map(float, coord_str.strip("()").split(", ")))
    elif coord_str is None:
        return None
    else:
        raise ValueError(f"Unsupported coordinate format: {coord_str}")

def distance_between_points(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def distance_to_boundary(point, segment_start, segment_end):
    x1, y1 = segment_start
    x2, y2 = segment_end
    line_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_squared == 0:
        return distance_between_points(point, segment_start), segment_start
    t = max(0, min(1, ((point[0] - x1) * (x2 - x1) + (point[1] - y1) * (y2 - y1)) / line_length_squared))
    projection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return distance_between_points(point, projection), projection

def distance_to_obstacle(point, vertices):
    min_distance = float('inf')
    closest_point = None
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        dist, closest = distance_to_boundary(point, v1, v2)
        if dist < min_distance:
            min_distance = dist
            closest_point = closest
    return min_distance, closest_point

def is_point_in_triangle(point, v1, v2, v3):
    px, py = point
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    if denominator == 0:  # Degenerate triangle (all points collinear)
        return False

    alpha = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denominator
    beta = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denominator
    gamma = 1 - alpha - beta

    return alpha >= 0 and beta >= 0 and gamma >= 0

def is_point_in_circle(point, center, radius):
    distance = distance_between_points(point, center)
    return distance <= radius, distance

def is_point_in_rectangle(point, bottom_left, top_right):
    px, py = point
    x0, y0 = bottom_left
    x1, y1 = top_right
    in_rectangle = x0 <= px <= x1 and y0 <= py <= y1
    if in_rectangle:
        return True, 0
    else:
        # Calculate the distance to the closest side
        dx = max(x0 - px, 0, px - x1)
        dy = max(y0 - py, 0, py - y1)
        return False, math.sqrt(dx ** 2 + dy ** 2)

def check_for_collision(point, row):
    row = row.squeeze()
    shape_type = row['cstr_obstacle_shape']
    midpoint = (row['cflt_midpoint_x_coord'], row['cflt_midpoint_y_coord'])
    bottom_left = parse_coordinates(row['cstr_bottom_left'])
    bottom_right = parse_coordinates(row['cstr_bottom_right'])
    top_right = parse_coordinates(row['cstr_top_right'])
    top_left = parse_coordinates(row['cstr_top_left'])
    mid_top = parse_coordinates(row['cstr_mid_top'])
    if shape_type == 'rect':
        in_shape, distance = is_point_in_rectangle(point, bottom_left, top_right)
        boundary_distance = distance
    elif shape_type == 'circle':
        radius = (top_right[0] - bottom_left[0])/2
        in_shape, boundary_distance  = is_point_in_circle(point, midpoint, radius)
        distance = 0 if in_shape else boundary_distance
    elif shape_type == 'polygon':
        vertices = [bottom_left, bottom_right, mid_top]
        in_shape = is_point_in_triangle(point, *vertices)
        boundary_distance, closest_point = distance_to_obstacle(point, vertices)
        distance = 0 if in_shape else boundary_distance
    else:
        return False, None, None, None, None

    midpoint_distance = distance_between_points(point, midpoint)
    exit_proximity = 1 - (boundary_distance/(boundary_distance + midpoint_distance)) if in_shape else 1.0
    return in_shape, distance, midpoint_distance, boundary_distance, exit_proximity
