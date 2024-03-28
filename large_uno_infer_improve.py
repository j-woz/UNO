import time
infer_start_time = time.time()
import os
import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# Additional imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# [Req] Imports from other scripts
from uno_preprocess_improve import preprocess_params
from uno_train_improve import metrics_list, train_params, ss_res, ss_tot
from uno_utils_improve import data_generator, batch_predict, print_duration, clean_arrays, check_array

filepath = Path(__file__).resolve().parent  # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------

# Custom objects for loading UNO model
def r2(y_true, y_pred):
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ss_res / ss_tot
custom_objects = {
    "ss_res": ss_res,
    "ss_tot": ss_tot,
    "r2": r2
}


# [Req]
def run(params: Dict):
    """Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create file names and load data for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    test_ge_fname = f"ge_{test_data_fname}"
    test_md_fname = f"md_{test_data_fname}"
    test_rsp_fname = f"rsp_{test_data_fname}"
    ts_ge = pd.read_parquet(Path(params["test_ml_data_dir"])/test_ge_fname)
    ts_md = pd.read_parquet(Path(params["test_ml_data_dir"])/test_md_fname)
    ts_rsp = pd.read_parquet(Path(params["test_ml_data_dir"])/test_rsp_fname)

    # Get real and predicted y_test and convert to numpy for compatibility
    # y_ts = ts_data[params["y_col_name"]].to_numpy()
    # x_ts = ts_data.drop([params["y_col_name"]], axis=1).to_numpy()

    # Test data generator
    generator_batch_size = params["generator_batch_size"]
    test_steps = int(np.ceil(len(ts_rsp) / generator_batch_size))
    test_gen = data_generator(ts_ge, ts_md, ts_rsp, generator_batch_size, params)
    

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"])  # [Req]

    # Load UNO
    model = load_model(modelpath, custom_objects=custom_objects)

    # Use batch_predict for predictions
    test_pred, test_true = batch_predict(
        model,
        data_generator(ts_ge, ts_md, ts_rsp, generator_batch_size, params, merge_preserve_order=True, verbose=False),
        test_steps
    )

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        outdir=params["infer_outdir"],
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    # Make sure test scores don't contain NANs
    test_pred_clean, test_true_clean = clean_arrays(test_pred, test_true)
    # Compute scores
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true_clean,
        y_pred=test_pred_clean,
        stage="test",
        outdir=params["infer_outdir"],
        metrics=metrics_list,
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        # default_model="params_ws.txt",
        # default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_params,
        required=None,
    )
    test_scores = run(params)
    infer_end_time = time.time()
    print_duration("Infering", infer_start_time, infer_end_time)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])