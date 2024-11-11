from app.data_models import tbl_model_runs, tbl_local_state, tbl_global_state, tbl_drone_actions, tbl_rewards, \
    tbl_model_run_params, tbl_map_data, tbl_rai
import datetime
import pandas as pd


pause_file_path = "/CapstoneTeamAPI/utils/status/model_pause_flag.txt"

# Lookup to run table creation functions
tbl_utilities = {"tbl_model_runs": tbl_model_runs,
                    "tbl_model_run_params": tbl_model_run_params,
                    "tbl_local_state": tbl_local_state,
                    "tbl_global_state": tbl_global_state,
                    "tbl_drone_actions": tbl_drone_actions,
                    "tbl_rewards": tbl_rewards,
                    "tbl_rai": tbl_rai,
                    "tbl_map_data": tbl_map_data}

# Lookup for batch sizes for table commits
tbl_batches = {"tbl_model_runs": 1,
                    "tbl_model_run_params": 1,
                    "tbl_local_state": 100,
                    "tbl_global_state": 1,
                    "tbl_drone_actions": 100,
                    "tbl_rewards": 100,
                    "tbl_rai": 100,
                    "tbl_map_data": 500}

# Function to scalarize table columns
def scalarize(pickled_table):
    tbl = {}
    for col in pickled_table.columns:
        if col[:5] == 'cint_':
            tbl[col] = [int(i) if not pd.isna(i) else 0 for i in pickled_table[col]]
        elif col[:5] == 'cdtm_':
            tbl[col] = [datetime.datetime.strptime(i, "%Y%m%d_%H%M_%S") for i in pickled_table[col]]
        elif col[:5] == 'cflt_':
            tbl[col] = [float(i) for i in pickled_table[col]]
        elif col[:5] == 'cbln_':
            tbl[col] = [bool(i) for i in pickled_table[col]]
        elif col[:5] == 'cstr_':
            tbl[col] = [str(i) for i in pickled_table[col]]
        else:
            tbl[col] = [str(i) for i in pickled_table[col]]
    return tbl
