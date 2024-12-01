import random as r
import pandas as pd
import numpy as np
import pygame
import math

# from pygame.draw.rect(self.screen, self.building_color, (x, y, 40, 40))
def rectangle(map, color, rect):
    return pygame.draw.rect(map, color, rect)

# from pygame.draw.circle(self.screen, self.human_color, (x, y), 10)
def circle(map, color, pos, radius):
    return pygame.draw.circle(map, color, pos, radius)

# from pygame.draw.polygon(self.screen, self.tree_color, [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)])
def triangle(map, color, pos):
    return pygame.draw.polygon(map, color, pos)

def triangle_points(point, length):
    br = [point[0]+length, point[1]]
    height = length**2 - (0.5*length)**2
    t = [point[0]+(0.5*length), point[1]+height]
    return point, br, t

def enforce_minimum_size(size, minimum=1):
    if isinstance(size, tuple):
        return tuple(max(s, minimum) for s in size)
    return max(size, minimum)

# Update
import math

def triangle_points(point, length):
    # Bottom left
    x1 = point[0]
    y1 = point[1]
    # Bottom right
    x2 = point[0] + length
    y2 = point[1]
    # Height
    height = (math.sqrt(3) / 2) * length
    # Top
    x3 = point[0] + (length / 2)
    y3 = point[1] + height

    # Calculate the centroid (midpoint of the triangle)
    midpoint_x = (x1 + x2 + x3) / 3
    midpoint_y = (y1 + y2 + y3) / 3

    return {
        "cstr_obstacle_shape": "polygon",
        "cstr_bottom_left": (x1, y1),
        "cstr_bottom_right": (x2, y2),
        "cstr_top_right": None,
        "cstr_top_left": None,
        "cstr_mid_top": (x3, y3),
        "cflt_midpoint_x_coord": midpoint_x,
        "cflt_midpoint_y_coord": midpoint_y
    }

def rectangle_points(point, v_length, h_length):
    # Bottom left
    x1 = point[0]
    y1 = point[1]
    # Bottom right
    x2 = point[0] + h_length
    y2 = point[1]
    # Top right
    x3 = point[0] + h_length
    y3 = point[1] + v_length
    # Top left
    x4 = point[0]
    y4 = point[1] + v_length

    # Calculate the midpoint of the rectangle
    midpoint_x = (x1 + x3) / 2
    midpoint_y = (y1 + y3) / 2

    return {
        "cstr_obstacle_shape": "rect",
        "cstr_bottom_left": (x1, y1),
        "cstr_bottom_right": (x2, y2),
        "cstr_top_right": (x3, y3),
        "cstr_top_left": (x4, y4),
        "cstr_mid_top": None,
        "cflt_midpoint_x_coord": midpoint_x,
        "cflt_midpoint_y_coord": midpoint_y
    }

def circle_points(point, radius):
    # Bottom left
    x1 = point[0] - radius
    y1 = point[1] - radius
    # Top right
    x3 = point[0] + radius
    y3 = point[1] + radius

    # The midpoint of a circle is its center
    midpoint_x = point[0]
    midpoint_y = point[1]

    return {
        "cstr_obstacle_shape": "circle",
        "cstr_bottom_left": (x1, y1),
        "cstr_bottom_right": None,
        "cstr_top_right": (x3, y3),
        "cstr_top_left": None,
        "cstr_mid_top": None,
        "cflt_midpoint_x_coord": midpoint_x,
        "cflt_midpoint_y_coord": midpoint_y
    }

class EnvironmentMap:
     def __init__(self,
                  map_size=[100, 100],
                  min_distance = 5,
                  target=dict(target_x_coordinate=20, target_y_coordinate=20),
                  no_fly_zones=dict(count=1, random=True, positions=[], sizes=[], damage=1),
                  humans=dict(count=1, random=True, positions=[], sizes=[], damage=1),
                  buildings=dict(count=1, random=True, positions=[], sizes=[], damage=1),
                  trees=dict(count=1, random=True, positions=[], sizes=[], damage=1),
                  animals=dict(count=1, random=True, positions=[], sizes=[], damage=1),
                  fires=dict(count=1, random=True, positions=[], sizes=[], damage=1)):

         self.map_size = map_size
         self.min_distance = min_distance
         self.map = pygame.Surface(self.map_size, pygame.SRCALPHA)
         self.colors = {'humans': (0, 0, 255),
                        'buildings': (128, 128, 128),
                        'trees': (0, 255, 0),
                        'animals': (139, 69, 19),
                        'no-fly': (255, 255, 0),
                        'fires':(255, 165, 0),
                        'target':(0,0,0)}

         self.obstacles = {'no-fly': no_fly_zones,
                           'humans': humans,
                           'buildings': buildings,
                           'trees': trees,
                           'animals': animals,
                           'fires': fires,
                           'target': target}

         self.scaler = {'no-fly': 1/5,
                           'humans': 1/8,
                           'buildings': 1/50,
                           'trees': 1/8,
                           'animals': 1/20,
                            'fires': 1/5,
                            'target':1/8}

         self.obstacle_func = {'no-fly': 'rectangle',
                                'humans': 'circle',
                                'buildings': 'rectangle',
                                'trees': 'triangle',
                                'animals': 'circle',
                                'fires': 'circle',
                               'target':'point'}
         self.dataframe = None


     def get_shape_data(self, id, obstacle):
         mask = pygame.mask.from_surface(self.map)
         boundary = mask.outline()
         interior = []
         for x in range(self.map_size[0]):
             for y in range(self.map_size[1]):
                 if mask.get_at((x, y)) and (x, y) not in boundary:
                     interior.append((x, y))
         all_points = boundary + interior
         all_x_coords = [point[0] for point in all_points]
         all_y_coords = [point[1] for point in all_points]

         if all_x_coords and all_y_coords:
             midpoint_x = int(np.mean(all_x_coords))
             midpoint_y = int(np.mean(all_y_coords))
         else:
             midpoint_x, midpoint_y = all_points[0][0], all_points[0][1]  # Handle empty case
         boundary_df = pd.DataFrame({
             'cint_obstacle_id':[id for pos in boundary],
             'cstr_obstacle':[obstacle for pos in boundary],
             'cflt_obstacle_risk':[self.obstacles[obstacle]["damage"] for pos in boundary],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in boundary],
             'cstr_point_type': ["boundary" for pos in boundary],
             'cflt_x_coord':[pos[0] for pos in boundary],
             'cflt_y_coord':[pos[1] for pos in boundary]
         })
         interior_df = pd.DataFrame({
             'cint_obstacle_id': [id for pos in interior],
             'cstr_obstacle': [obstacle for pos in interior],
             'cflt_obstacle_risk': [self.obstacles[obstacle]["damage"] for pos in interior],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in interior],
             'cstr_point_type': ["interior" for pos in interior],
             'cflt_x_coord': [pos[0] for pos in interior],
             'cflt_y_coord': [pos[1] for pos in interior]
         })

         midpoint_df = pd.DataFrame({
             'cint_obstacle_id': [id],
             'cstr_obstacle': [obstacle],
             'cflt_obstacle_risk': [self.obstacles[obstacle]["damage"]],
             'cstr_obstacle_color': [],
             'cstr_point_type': ["midpoint"],
             'cflt_x_coord': [midpoint_x],
             'cflt_y_coord': [midpoint_y]
         })

         if self.dataframe is None:
            self.dataframe = pd.concat([midpoint_df, interior_df, boundary_df])
         else:
            self.dataframe = pd.concat([self.dataframe, midpoint_df, interior_df, boundary_df])
         self.map.fill((0, 0, 0, 0))

     def get_shape_data2(self, id, obstacle, coord_data):
        print(f"MAP SIZE {self.map_size}")
        if obstacle == 'target':
            add_df = pd.DataFrame({
                'cint_obstacle_id': [0],
                'cstr_obstacle': [obstacle],
                'cflt_obstacle_risk': [0],
                'cstr_obstacle_color': [self.colors[obstacle]],
                'cstr_obstacle_shape': ['point'],
                'cflt_midpoint_x_coord': [self.obstacles[obstacle]['target_x_coordinate']],
                'cflt_midpoint_y_coord': [self.obstacles[obstacle]['target_y_coordinate']],
                "cstr_bottom_left": [None],
                "cstr_bottom_right": [None],
                "cstr_top_right": [None],
                "cstr_top_left": [None],
                "cstr_mid_top": [None]
            })
        else:
            add_df = pd.DataFrame({
                'cint_obstacle_id': [id],
                'cstr_obstacle': [obstacle],
                'cflt_obstacle_risk': [self.obstacles[obstacle]["damage"]],
                'cstr_obstacle_color': [self.colors[obstacle]],
                'cstr_obstacle_shape': [coord_data["cstr_obstacle_shape"]],
                'cflt_midpoint_x_coord': [coord_data["cflt_midpoint_x_coord"]],
                'cflt_midpoint_y_coord': [coord_data["cflt_midpoint_y_coord"]],
                "cstr_bottom_left": [coord_data["cstr_bottom_left"]],
                 "cstr_bottom_right": [coord_data["cstr_bottom_right"]],
                 "cstr_top_right": [coord_data["cstr_top_right"]],
                 "cstr_top_left": [coord_data["cstr_top_left"]],
                 "cstr_mid_top": [coord_data["cstr_mid_top"]]
            })
        if self.dataframe is None:
            self.dataframe = add_df
        else:
            self.dataframe = pd.concat([self.dataframe, add_df])

     def generate_random_position(self, existing_positions):
         for i in range(100):  # Attempt up to 100 times
             x = r.randint(0, self.map_size[0] - 1)
             y = r.randint(0, self.map_size[1] - 1)
             if all(math.hypot(x - ex, y - ey) >= self.min_distance for ex, ey in existing_positions):
                 return (x, y)

     def obstacle_coords(self, obstacle, rg):
         self.map.fill((0, 0, 0, 0))
         if self.obstacle_func[obstacle] == 'circle':
             self.obstacles[obstacle]["radius"] = [enforce_minimum_size(r.randint(0, 50)*self.scaler[obstacle]) for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["radius"])
             print(f"Radius: {self.obstacles[obstacle]['radius']}")
             for i, el in enumerate(zipper, start=1):
                 circle(self.map, self.colors[obstacle], el[0], el[1])
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'rectangle':
             l = r.randint(0, self.map_size[0]*self.scaler[obstacle])
             w = r.randint(0, self.map_size[0]*self.scaler[obstacle])
             self.obstacles[obstacle]["wh"] = [enforce_minimum_size((r.randint(0, 50), r.randint(0, 50))) for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["wh"])
             for i, el in enumerate(zipper, start=1):
                 rectangle(self.map, self.colors[obstacle], (el[0][0], el[0][1], el[1][0], el[1][1]))
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'triangle':
             l = r.randint(0, round(self.map_size[0] * self.scaler[obstacle],0))
             self.obstacles[obstacle]["tri"] = [triangle_points(point, l) for point in self.obstacles[obstacle]["positions"]]
             for i, el in enumerate(self.obstacles[obstacle]["tri"], start=1):
                 rectangle(self.map, self.colors[obstacle], (el[0][0], el[0][1], el[1][0], el[1][1]))
                 self.get_shape_data(i, obstacle)


    # Update
     def obstacle_coords2(self, obstacle, rg):
         if self.obstacle_func[obstacle] == 'circle':
             l = r.uniform(0, self.map_size[0] * self.scaler[obstacle])
             pos = self.obstacles[obstacle]["positions"]
             #print(f"Circle Radius: {l}, Circle Positions: {pos}")
             for i, el in enumerate(pos, start=1):
                 coords = circle_points(el, l)
                 self.get_shape_data2(i, obstacle, coords)


         elif self.obstacle_func[obstacle] == 'rectangle':
             l = r.uniform(1, self.map_size[0]*self.scaler[obstacle])
             w = r.uniform(1, self.map_size[0]*self.scaler[obstacle])
             pos = self.obstacles[obstacle]["positions"]
             #print(f"Rectangle Width: {w}, Rectangle Length:{l}, Rectangle Positions: {pos}")
             for i, el in enumerate(pos, start=1):
                 coords = rectangle_points(el, l, w)
                 self.get_shape_data2(i, obstacle, coords)

         elif self.obstacle_func[obstacle] == 'triangle':
             l = r.uniform(0, round(self.map_size[0] * self.scaler[obstacle],0))
             pos = self.obstacles[obstacle]["positions"]
             #print(f"Triangle Length: {l}, Triangle Positions: {pos}")
             for i, el in enumerate(pos, start=1):
                 coords = triangle_points(el, l)
                 self.get_shape_data2(i, obstacle, coords)

     def generate_obstacle_data(self):
         for key, val in self.obstacles.items():
             # If all attributes have been entered
             if key == 'target':
                 self.get_shape_data2(None, key, None)
             else:
                 if all(k in val for k in ["count", "positions", "damage", "random"]):
                     rg = range(val["count"])
                 else:
                     print(f"Error: Missing obstacle attributes: {key}")
                     continue
                 # If the count of the obstacle is greater than zero
                 if val["count"] > 0:
                     existing_positions = []
                     # If random positions = True
                     if self.obstacles[key]["random"]:
                         obs_pos = []
                         for i in rg:
                             pos = self.generate_random_position(existing_positions)
                             if pos:
                                 obs_pos.append(pos)
                                 existing_positions.append(pos)
                         self.obstacles[key]["positions"] = obs_pos
                     self.obstacle_coords2(key, rg)
         if self.dataframe is None:
             self.dataframe = pd.DataFrame(columns=[
             'cint_obstacle_id',
             'cstr_obstacle',
             'cflt_obstacle_risk',
             'cstr_obstacle_color',
             'cstr_point_type',
             'cflt_x_coord',
             'cflt_y_coord'
         ])


import plotly.graph_objects as go


def parse_coordinates(coord_str):
    """
    Converts a string like "(4, 5)" to a tuple (4, 5).
    """
    if isinstance(coord_str, tuple):
        return coord_str
    return tuple(map(float, coord_str.strip("()").split(", ")))


def plot_obstacles(obstacle_data):
    """
    Graphs obstacles using add_shape based on the provided data.

    Arguments:
    obstacle_data -- A DataFrame containing obstacle data with columns:
                     ['cint_obstacle_id', 'cstr_obstacle', 'cflt_obstacle_risk',
                      'cstr_obstacle_color', 'cstr_obstacle_shape', 'cflt_midpoint_x_coord',
                      'cflt_midpoint_y_coord', 'cstr_bottom_left', 'cstr_bottom_right',
                      'cstr_top_right', 'cstr_top_left', 'cstr_mid_top']
    """

    def convert_color_to_rgba(color, alpha=0.5):
        """
        Converts a tuple color (R, G, B) to an rgba string.
        """
        return f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"

    # Create the figure
    fig = go.Figure()

    # Loop through each obstacle data entry
    for _, row in obstacle_data.iterrows():
        color = convert_color_to_rgba(row['cstr_obstacle_color'])  # Color of the obstacle
        shape_type = row['cstr_obstacle_shape']  # Shape of the obstacle
        x_center = row['cflt_midpoint_x_coord']  # Midpoint X coordinate
        y_center = row['cflt_midpoint_y_coord']  # Midpoint Y coordinate

        # Handle each shape type accordingly
        if shape_type == 'rect':
            # Convert string coordinates to tuples
            bottom_left = parse_coordinates(row['cstr_bottom_left'])
            top_right = parse_coordinates(row['cstr_top_right'])

            x0, y0 = bottom_left  # Bottom-left corner
            x1, y1 = top_right  # Top-right corner

            # Add rectangle shape
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0,
                x1=x1, y1=y1,
                line=dict(color=color, width=2),
                fillcolor=color,
                name=f"Obstacle {row['cint_obstacle_id']}"
            )

        elif shape_type == 'circle':
            # If it's a circle, we can assume the midpoint as the center
            radius = row['cflt_obstacle_risk']  # We can assume that the damage might indicate the radius

            # Add circle shape (circle requires x0, y0, x1, y1 for bounding box)
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x_center - radius, y0=y_center - radius,
                x1=x_center + radius, y1=y_center + radius,
                line=dict(color=color, width=2),
                fillcolor=color,
                name=f"Obstacle {row['cint_obstacle_id']}"
            )


        elif shape_type == 'polygon':

            # Convert string coordinates to tuples

            bottom_left = parse_coordinates(row['cstr_bottom_left'])

            bottom_right = parse_coordinates(row['cstr_bottom_right'])

            mid_top = parse_coordinates(row['cstr_mid_top'])

            x_coords = [bottom_left[0], bottom_right[0], mid_top[0], bottom_left[0]]

            y_coords = [bottom_left[1], bottom_right[1], mid_top[1], bottom_left[1]]

            # Add Scatter trace for polygon

            fig.add_trace(go.Scatter(

                x=x_coords,

                y=y_coords,

                fill="toself",

                fillcolor=color,

                line=dict(color=color, width=2),

                mode="lines"

            ))

    # Update layout
    fig.update_layout(
        title="Obstacle Map",
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(scaleanchor="x"),
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=30),
        height=600,
    )

    # Show the figure
    fig.show()


# tst = EnvironmentMap()
# tst.generate_obstacle_data()
# tst.dataframe.columns
# plot_obstacles(tst.dataframe)