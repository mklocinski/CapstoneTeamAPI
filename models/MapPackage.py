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

class EnvironmentMap:
     def __init__(self,
                  map_size=[100, 100],
                  min_distance = 5,
                  no_fly_zones=dict(count=1, random=True, positions=[], sizes=[], risk=1),
                  humans=dict(count=1, random=True, positions=[], sizes=[], risk=1),
                  buildings=dict(count=1, random=True, positions=[], sizes=[], risk=1),
                  trees=dict(count=1, random=True, positions=[], sizes=[], risk=1),
                  animals=dict(count=1, random=True, positions=[], sizes=[], risk=1)):

         self.map_size = map_size
         self.min_distance = min_distance
         self.map = pygame.Surface(self.map_size, pygame.SRCALPHA)
         self.colors = {'humans': (255, 165, 0),
                        'buildings': (128, 128, 128),
                        'trees': (0, 255, 0),
                        'animals': (139, 69, 19),
                        'no-fly': (255, 255, 0)}

         self.obstacles = {'no-fly': no_fly_zones,
                           'humans': humans,
                           'buildings': buildings,
                           'trees': trees,
                           'animals': animals}

         self.scaler = {'no-fly': 1/5,
                           'humans': 1/100,
                           'buildings': 1/50,
                           'trees': 1/20,
                           'animals': 1/20}

         self.obstacle_func = {'no-fly': 'rectangle',
                                'humans': 'circle',
                                'buildings': 'rectangle',
                                'trees': 'triangle',
                                'animals': 'circle'}
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
             midpoint_x, midpoint_y = None, None  # Handle empty case
         boundary_df = pd.DataFrame({
             'cint_obstacle_id':[id for pos in boundary],
             'cstr_obstacle':[obstacle for pos in boundary],
             'cflt_obstacle_risk':[self.obstacles[obstacle]["risk"] for pos in boundary],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in boundary],
             'cstr_point_type': ["boundary" for pos in boundary],
             'cflt_x_coord':[pos[0] for pos in boundary],
             'cflt_y_coord':[pos[1] for pos in boundary]
         })
         interior_df = pd.DataFrame({
             'cint_obstacle_id': [id for pos in interior],
             'cstr_obstacle': [obstacle for pos in interior],
             'cflt_obstacle_risk': [self.obstacles[obstacle]["risk"] for pos in interior],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in interior],
             'cstr_point_type': ["boundary" for pos in interior],
             'cflt_x_coord': [pos[0] for pos in interior],
             'cflt_y_coord': [pos[1] for pos in interior]
         })

         midpoint_df = pd.DataFrame({
             'cint_obstacle_id': [id],
             'cstr_obstacle': [obstacle],
             'cflt_obstacle_risk': [self.obstacles[obstacle]["risk"]],
             'cstr_obstacle_color': [self.colors[obstacle]],
             'cstr_point_type': ["midpoint"],
             'cflt_x_coord': [midpoint_x],
             'cflt_y_coord': [midpoint_y]
         })

         if self.dataframe is None:
             self.dataframe = pd.concat([midpoint_df, interior_df, boundary_df])
         else:
             self.dataframe = pd.concat([self.dataframe, midpoint_df, interior_df, boundary_df])
         self.map.fill((0, 0, 0, 0))

     def generate_random_position(self, existing_positions):
         for i in range(100):  # Attempt up to 100 times
             x = r.randint(0, self.map_size[0] - 1)
             y = r.randint(0, self.map_size[1] - 1)
             if all(math.hypot(x - ex, y - ey) >= self.min_distance for ex, ey in existing_positions):
                 return (x, y)

     def obstacle_coords(self, obstacle, rg):
         self.map.fill((0, 0, 0, 0))
         if self.obstacle_func[obstacle] == 'circle':
             self.obstacles[obstacle]["radius"] = [r.randint(0, 50)*self.scaler[obstacle] for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["radius"])
             for i, el in enumerate(zipper, start=1):
                 circle(self.map, self.colors[obstacle], el[0], el[1])
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'rectangle':
             l = r.randint(0, self.map_size[0]*self.scaler[obstacle])
             w = r.randint(0, self.map_size[0]*self.scaler[obstacle])
             self.obstacles[obstacle]["wh"] = [(r.randint(0, 50), r.randint(0, 50)) for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["wh"])
             for i, el in enumerate(zipper, start=1):
                 rectangle(self.map, self.colors[obstacle], (el[0][0], el[0][1], el[1][0], el[1][1]))
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'triangle':
             l = r.randint(0, self.map_size[0] * self.scaler[obstacle])
             self.obstacles[obstacle]["tri"] = [triangle_points(point, l) for point in self.obstacles[obstacle]["positions"]]
             for i, el in enumerate(self.obstacles[obstacle]["tri"], start=1):
                    triangle(self.map, self.colors[obstacle], el)
                    self.get_shape_data(i, obstacle)

     def generate_obstacle_data(self):
         for key, val in self.obstacles.items():
             # If all attributes have been entered
             if all(k in val for k in ["count", "positions", "risk", "random"]):
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
                 self.obstacle_coords(key, rg)


