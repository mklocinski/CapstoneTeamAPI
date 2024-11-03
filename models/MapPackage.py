import random as r
import pandas as pd
import pygame

# from pygame.draw.rect(self.screen, self.building_color, (x, y, 40, 40))
def rectangle(map, color, rect):
    return pygame.draw.rect(map, color, rect)

# from pygame.draw.circle(self.screen, self.human_color, (x, y), 10)
def circle(map, color, pos, radius):
    return pygame.draw.circle(map, color, pos, radius)

# from pygame.draw.polygon(self.screen, self.tree_color, [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)])
def triangle(map, color, pos):
    return pygame.draw.polygon(map, color, pos)



class EnvironmentMap:
     def __init__(self,
                  map_size=[100, 100],
                  no_fly_zones=dict(count=5, random=True, positions=[], sizes=[], risk=1),
                  humans=dict(count=5, random=True, positions=[], sizes=[], risk=1),
                  buildings=dict(count=5, random=True, positions=[], sizes=[], risk=1),
                  trees=dict(count=5, random=True, positions=[], sizes=[], risk=1),
                  animals=dict(count=5, random=True, positions=[], sizes=[], risk=1)):

         self.map_size = map_size
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
         boundary_df = pd.DataFrame({
             'cint_obstacle_id':[id for pos in boundary],
             'cstr_obstacle':[obstacle for pos in boundary],
             'cint_obstacle_risk':[self.obstacles[obstacle]["risk"] for pos in boundary],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in boundary],
             'cstr_point_type': ["boundary" for pos in boundary],
             'cflt_x_coord':[pos[0] for pos in boundary],
             'cflt_y_coord':[pos[1] for pos in boundary]
         })
         interior_df = pd.DataFrame({
             'cint_obstacle_id': [id for pos in interior],
             'cstr_obstacle': [obstacle for pos in interior],
             'cint_obstacle_risk': [self.obstacles[obstacle]["risk"] for pos in interior],
             'cstr_obstacle_color': [self.colors[obstacle] for pos in interior],
             'cstr_point_type': ["boundary" for pos in interior],
             'cflt_x_coord': [pos[0] for pos in interior],
             'cflt_y_coord': [pos[1] for pos in interior]
         })
         if self.dataframe is None:
             self.dataframe = pd.concat([interior_df, boundary_df])
         else:
             self.dataframe = pd.concat([self.dataframe, interior_df, boundary_df])
         self.map.fill((0, 0, 0, 0))

     def obstacle_coords(self, obstacle, rg):
         self.map.fill((0, 0, 0, 0))

         if self.obstacle_func[obstacle] == 'circle':
             self.obstacles[obstacle]["radius"] = [r.randint(0, 50) for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["radius"])
             for i, el in enumerate(zipper, start=1):
                 circle(self.map, self.colors[obstacle], el[0], el[1])
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'rectangle':
             self.obstacles[obstacle]["wh"] = [(r.randint(0, 50), r.randint(0, 50)) for i in rg]
             zipper = zip(self.obstacles[obstacle]["positions"], self.obstacles[obstacle]["wh"])
             for i, el in enumerate(zipper, start=1):
                 rectangle(self.map, self.colors[obstacle], (el[0][0], el[0][1], el[1][0], el[1][1]))
                 self.get_shape_data(i, obstacle)

         elif self.obstacle_func[obstacle] == 'triangle':
             self.obstacles[obstacle]["tri"] = [[(r.randint(0, 50), r.randint(0, 50)) for j in range(3)] for i in rg]
             for i, el in enumerate(self.obstacles[obstacle]["tri"], start=1):
                    triangle(self.map, self.colors[obstacle], el)
                    self.get_shape_data(i, obstacle)

     def generate_obstacle_data(self):
         for key, val in self.obstacles.items():
             rg = range(self.obstacles[key]["count"])
             if self.obstacles[key]["random"]:
                 self.obstacles[key]["sizes"] = [r.randint(0, 50) for i in rg]
                 self.obstacles[key]["positions"] = [(r.randint(0, 50), r.randint(0, 50)) for s in
                                                     self.obstacles[key]["sizes"]]
             self.obstacle_coords(key, rg)


