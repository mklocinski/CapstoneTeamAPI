from app.data_models import tbl_model_runs, tbl_local_state, tbl_global_state, tbl_drone_actions, tbl_rewards, \
    tbl_model_run_params, tbl_map_data, tbl_rai
import datetime

tbl_utilities = {"tbl_model_runs": tbl_model_runs,
                    "tbl_model_run_params": tbl_model_run_params,
                    "tbl_local_state": tbl_local_state,
                    "tbl_global_state": tbl_global_state,
                    "tbl_drone_actions": tbl_drone_actions,
                    "tbl_rewards": tbl_rewards,
                    "tbl_rai": tbl_rai,
                    "tbl_map_data": tbl_map_data}

def scalarize(pickled_table):
    tbl = {}
    for col in pickled_table.columns:
        if col[:5] == 'cint_':
            tbl[col] = [int(i) for i in pickled_table[col]]
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
