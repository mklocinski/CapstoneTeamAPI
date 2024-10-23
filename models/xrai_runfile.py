import sys
import os

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = project_root

import WrappedModel as wm
import TestingMapPackage as mp
import json


# -----------------------------------------------------------------------------#
# -------------------------------- Run System ---------------------------------#
# -----------------------------------------------------------------------------#


def main(environment_inputs,
                    model_inputs,
                    map_inputs,
                    current_state):

    print("Making map...")
    # -------------------------------- Create Map ---------------------------------#
    map = mp.MapMaker(map_size=[environment_inputs["world_size"], environment_inputs["world_size"]])
    map.initialize_empty_map(),
    map.generate_random_circles(map_inputs["obstacle1"])
    map.get_all_coordinates()

    # ------------------------------------ Run ------------------------------------#

    environment_id = environment_inputs["environment_id"]
    environment_inputs.pop("environment_id")
    environment_inputs["comm_radius"] = 100* np.sqrt(environment_inputs["comm_radius"])
    print("Running model...")
    wm.main(environment_id, environment_inputs, model_inputs, map, current_state)








if __name__ == '__main__':

    print("Getting passed arguments...")
    environment_inputs = sys.argv[1]
    model_inputs = sys.argv[2]
    map_inputs = sys.argv[3]

    if len(sys.argv)==5:
        current_state = sys.argv[4]
    else:
        current_state = None

    environment_inputs = json.loads(environment_inputs)
    model_inputs = json.loads(model_inputs)
    map_inputs = json.loads(map_inputs)

    main(environment_inputs, model_inputs, map_inputs, current_state=current_state)


