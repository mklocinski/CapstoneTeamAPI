from deep_rl_for_swarms import run_multiagent_trpo
from DRLOutput import DRLOutput, FlattenOutput


# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# Main Sections:
# > Run Model
#   > Runs the DRLSS model with the DRLOutput wrapper. Later versions will include
#      RAI and XAI wrappers
# Convert to Output to Dataframes
#   > Converts JSON from DRLOutput wrapper to a list of DataFrames to be
#       converted into a database (\app\routes.py).

# -------------------------------------------------------------------------- #
# ------------------------------ Run Model --------------------------------- #
# -------------------------------------------------------------------------- #

model = run_multiagent_trpo.main()



