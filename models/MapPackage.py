import random

class EnvironmentMap:
    def __init__(self, map_size=[100, 100], num_no_fly_zones=3, num_humans=5, num_buildings=5, num_trees=5, num_animals=5):
        self.window_width = map_size[0]
        self.window_height = map_size[1]
        #self.drone_radius = drone_radius  # Radius is half of the diameter (5 pixels)
        #self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        #pygame.display.set_caption("Drone Swarm Visualization")
        #self.font = pygame.font.SysFont(None, 16)
        #self.clock = pygame.time.Clock()
        #self.fps = 30  # Frames per second

        # Initialize obstacle lists and colors for the obstacles/drones.
        self.drone_color = (0, 0, 255)  # Blue for drones
        self.human_color = (255, 165, 0)  # Orange color for humans
        self.building_color = (128, 128, 128)  # Gray color for buildings
        self.tree_color = (0, 255, 0)  # Green color for trees
        self.animal_color = (139, 69, 19)  # Brown color for animals
        self.humans = []
        self.buildings = []
        self.trees = []
        self.animals = []

        # Initialize no-fly zones
        self.no_fly_zones = []
        self.num_no_fly_zones = num_no_fly_zones
        self.generate_no_fly_zones(self.num_no_fly_zones)
        self.no_fly_zone_color = (255, 255, 0)  # Yellow

        # Generate a random target location
        self.target_location = self.generate_random_target()
        self.target_radius = 10  # Radius of the target


    def generate_no_fly_zones(self, num_no_fly_zones):
        # Generate random positions and sizes for no-fly zones
        for i in range(num_no_fly_zones):
            size = random.randint(50, 150)  # Random size between 50 and 150 pixels
            x = random.randint(0, self.window_width - size)
            y = random.randint(0, self.window_height - size)
            self.no_fly_zones.append({"id": i, "position": (x, y), "size": size})


    def draw_no_fly_zones(self):
        # Draw the no-fly zones as yellow rectangles
        for zone in self.no_fly_zones:
            x, y = zone['position']
            size = zone['size']
            pygame.draw.rect(self.screen, self.no_fly_zone_color, (x, y, size, size), width=2)

            # Label no fly zone with its ID
            label = self.font.render(f"No Fly Zone {zone['id']}", True, (0, 0, 0))  # Black label
            label_rect = label.get_rect(center=(x + size // 2, y + size // 2))  # Center the label inside the rectangle
            self.screen.blit(label, label_rect)


    def generate_random_target(self):
        # Generate a random target location in the environment.
        target_x = random.randint(0, 100)  # Assuming the environment size is 100x100
        target_y = random.randint(0, 100)
        return target_x, target_y


    def generate_obstacles(self, num_humans, num_buildings, num_trees, num_animals):
        # Generates random positions for different types of obstacles.
        self.humans = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_humans)]
        self.buildings = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_buildings)]
        self.trees = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_trees)]
        self.animals = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_animals)]


    def draw_obstacles(self):
        # Draws obstacles (humans, buildings, trees, and animals) on the screen.
        # Draw humans (circles)
        for pos in self.humans:
            x, y = int((pos[0] / 100) * self.window_width), int((pos[1] / 100) * self.window_height)
            pygame.draw.circle(self.screen, self.human_color, (x, y), 10)
            label = self.font.render("Human", True, self.human_color)
            self.screen.blit(label, (x + 10, y))

        # Draw buildings (rectangles)
        for pos in self.buildings:
            x, y = int((pos[0] / 100) * self.window_width), int((pos[1] / 100) * self.window_height)
            pygame.draw.rect(self.screen, self.building_color, (x, y, 40, 40))  # Rectangle for building
            label = self.font.render("Building", True, self.building_color)
            self.screen.blit(label, (x + 5, y - 15))

        # Draw trees (triangles)
        for pos in self.trees:
            x, y = int((pos[0] / 100) * self.window_width), int((pos[1] / 100) * self.window_height)
            pygame.draw.polygon(self.screen, self.tree_color, [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)])
            label = self.font.render("Tree", True, self.tree_color)
            self.screen.blit(label, (x + 10, y))

        # Draw animals (small circles)
        for pos in self.animals:
            x, y = int((pos[0] / 100) * self.window_width), int((pos[1] / 100) * self.window_height)
            pygame.draw.circle(self.screen, self.animal_color, (x, y), 7)
            label = self.font.render("Animal", True, self.animal_color)
            self.screen.blit(label, (x + 10, y))


    def draw_drones_and_target(self, drone_positions):
        # Clear the screen with white background
        self.screen.fill((255, 255, 255))

        # Draw the target location as a red circle
        target_x, target_y = self.target_location
        target_x = int((target_x / 100) * self.window_width)
        target_y = int((target_y / 100) * self.window_height)
        pygame.draw.circle(self.screen, (255, 0, 0), (target_x, target_y), self.target_radius)

        # Draw the obstacles (humans, buildings, trees, and animals)
        self.draw_obstacles()
        # Draw no-fly zones
        self.draw_no_fly_zones()

        # Label the drone with its ID
        target_label = self.font.render(f"Target Location", True, (255, 0, 0))  # Red text for ID label
        self.screen.blit(target_label,
                         (target_x + 15, target_y - 10))  # Offset the label slightly to the right of the drone

        for i, pos in enumerate(drone_positions):
            # Extract x and y from the position
            if len(pos) >= 2:
                x, y = pos[:2]  # Get the first two elements (x, y)
            else:
                raise ValueError(f"Invalid drone position: {pos}")

            # Convert drone positions from environment coordinates to window coordinates
            x = int((x / 100) * self.window_width)  # Assume the environment size is 100x100
            y = int((y / 100) * self.window_height)

            # Draw the drone as a blue circle
            pygame.draw.circle(self.screen, self.drone_color, (x, y), self.drone_radius)

            # Label the drone with its ID
            label = self.font.render(f"Drone {i}", True, (0, 0, 0))  # Black text for ID label
            self.screen.blit(label, (x + 10, y))  # Offset the label slightly to the right of the drone

        pygame.display.update()