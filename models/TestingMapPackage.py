import numpy as np
import pandas as pd
import random as r

def euclidean_distance(x1, x2, y1, y2):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

class MapMaker:
    def __init__(self, map_size=[100,100]):
        #self.obstacle_dictionary = obstacle_values
        self.map_size = map_size
        self.map_matrix = None
        self.map_all_coordinates_df = None
        self.map_obstacle_coord_list = None
        self.map_obstacle_dictionary = None

    def initialize_empty_map(self):
        self.map_matrix = np.zeros(self.map_size)

    def generate_random_circles(self, n, max_pct_of_map=0.25):
        x = np.arange(self.map_size[1])
        y = np.arange(self.map_size[0])
        xs, ys = np.meshgrid(x, y)


        graph_data = [(r.sample(range(self.map_size[0]),1)[0],
                        r.sample(range(self.map_size[1]),1)[0],
                        r.sample(range(round(min(self.map_size)*max_pct_of_map)),1)[0]) for i in range(n)]

        for circle in graph_data:
            distance = euclidean_distance(xs, circle[0], ys, circle[1])
            self.map_matrix[distance <= circle[2]] = 1

    def get_all_coordinates(self):
        obstacle_points = np.where(self.map_matrix ==1)
        obstacle_df = pd.DataFrame({'x_coord':[x for x in obstacle_points[1]],
                                    'y_coord':[y for y in obstacle_points[0]],
                                    'obstacle': 1})
        non_obstacle_points = np.where(self.map_matrix == 0)
        non_obstacle_df = pd.DataFrame({'x_coord': [x for x in non_obstacle_points[1]],
                                    'y_coord': [y for y in non_obstacle_points[0]],
                                    'obstacle': 0})
        self.map_all_coordinates_df = pd.concat([obstacle_df, non_obstacle_df])

    def get_obstacle_coordinates(self):
        obstacle_points = np.where(self.map_matrix ==1)
        self.map_obstacle_coord_list = [(x,y) for x, y in zip(obstacle_points[1], obstacle_points[0])]




