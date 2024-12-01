import sys
import os
import numpy as np
import json
import pandas as pd
# Removed manual project root manipulation, since Heroku usually installs everything correctly within a defined environment
import XRAI_Model as wm


# -----------------------------------------------------------------------------#
# -------------------------------- Run System ---------------------------------#
# -----------------------------------------------------------------------------#

def main(environment_inputs, model_inputs, map_inputs, rai_inputs, current_state):

    # ------------------------------------ Run ------------------------------------#
    print(environment_inputs)
    print(type(environment_inputs))
    environment_id = environment_inputs["environment_id"]
    environment_inputs.pop("environment_id")
    environment_inputs["comm_radius"] = 100 * np.sqrt(environment_inputs["comm_radius"])

    print("Running model...")
    wm.main(environment_id, environment_inputs, model_inputs, map_inputs, rai_inputs, current_state)


if __name__ == '__main__':
    print(">> Runfile")
    print(">> >> Reading in parameters...")
    if len(sys.argv) >= 5:
        print(sys.argv[1])
        environment_inputs = json.loads(sys.argv[1])
        print(f">> >> Parameter Check - Environment Inputs: {environment_inputs}, type is {type(environment_inputs)}")
        model_inputs = json.loads(sys.argv[2])
        print(f">> >> Parameter Check - Model Inputs: {model_inputs}, type is {type(model_inputs)}")

        # Load map_inputs as JSON if it’s a path to a file, or as a DataFrame directly if it’s a JSON string
        map_inputs_str = sys.argv[3]

        if map_inputs_str.endswith('.json'):
            map_inputs = pd.read_json(map_inputs_str)
        else:
            map_inputs = pd.DataFrame(json.loads(map_inputs_str))
        print(f">> >> Parameter Check - Map Inputs: {map_inputs.head(1)}, type is {type(map_inputs)}")

        rai_inputs = json.loads(sys.argv[4])
        print(f">> >> Parameter Check - RAI Inputs: {rai_inputs}, type is {type(rai_inputs)}")

        # Handle optional current_state argument
        if len(sys.argv) == 6:
            current_state = json.loads(sys.argv[5])
        else:
            current_state = [None, None]
        print(f">> >> Parameter Check - Current State Inputs: {current_state[0]}, type is {type(current_state[0])}")

        # Call main function
        print(">> >> Calling main...")
        main(environment_inputs, model_inputs, map_inputs, rai_inputs, current_state)
    else:
        # Handle cases where the script is run without enough arguments
        print("Error: Not enough arguments provided.")
