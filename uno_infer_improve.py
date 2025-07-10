
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict
import traceback

# Import required modules from improvelib
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm  # Utility functions

# Additional third-party library imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import custom modules from local scripts
from uno_preprocess_improve import preprocess_params
from uno_train_improve import metrics_list, train_params
from uno_utils_improve import (
    data_merge_generator, batch_predict, print_duration, clean_arrays,
    check_array, calculate_sstot
)

from mae_poly_loss import mae_poly_loss
from uno_train_improve import import_custom_loss_fn, read_df

# Set filepath to the directory where the script is located
filepath = Path(__file__).resolve().parent  

# ---------------------
# Parameter Lists
# ---------------------
# Define two parameter lists required by the inference process:
# 1. App-specific parameters for monotherapy drug response prediction.
# 2. Model-specific parameters (optional; LightGBM in this case).

# Currently no app-specific parameters.
app_infer_params = []

# Optional model-specific parameters.
model_infer_params = []

# Combine both parameter lists to pass to frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params

# # data_cache[train|val|test|merged][ge|md|rsp] = None|DataFrame
# data_cache = None

merged_ge = None
merged_md = None


def run(params: Dict):
    """
    Run model inference and compute prediction scores.

    Args:
        params (dict): Dictionary containing model and application parameters.

    Returns:
        bool: True if inference completes successfully.
    """

    global merged_ge, merged_md
    
    # Read test RSPs (small)
    test_rsp = read_df(params, stage="test", label="rsp")
    print("infer: run(): test_rsp:   " + str(test_rsp))

    # Read merged GE, MD if needed (large)
    if merged_ge is None:
        merged_ge = read_df(params, stage="merged", label="ge")
        print("infer: run(): merged_ge:   " + str(merged_ge))
    if merged_md is None:
        merged_md = read_df(params, stage="merged", label="md")
        print("infer: run(): merged_md:   " + str(merged_md))
    
    model = do_load_model(params)
    
    do_infer(params, model, merged_ge, merged_md, test_rsp)


# def data_cache_init():
#     """ Conditionally initialize and return data_cache """
#     global data_cache
#     if data_cache is not None: return data_cache
    
#     data_cache = {}
#     for key1 in ["train", "val", "test", "merged"]:
#         data_cache[key1] = {}
#         for key2 in ["ge", "md", "rsp"]:
#             data_cache[key2] = None
#     return data_cache


# def load_merged(data_cache):
#     for label in ["ge", "md"]:
#         if data_cache["merged"][label] is None:
#             data_cache["merged"][label] = \
#                 read_df(params, stage="merged", label)

    
def do_load_model(params):
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build the model path
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    )
    # Load the pre-trained model
    # model = load_model(modelpath, compile=False)
    # model.compile(optimizer = "Adam", loss = "mse")
    print("loading model: '%s'" % modelpath)
   
    try:
        loss_function = "mse"
        if "custom_loss_module" in params and \
           params["custom_loss_module"] is not None:
           loss_function = import_custom_loss_fn(params)
        model = load_model(modelpath, compile=False)
        model.compile(optimizer = "Adam", loss = loss_function)
    except IOError as e:
        print("model load failed: " + str(e))
        exit(1)
        
    return model

    
def do_infer(params, model, ge, md, rsp):

    print("do_infer(): RSPs: %i ..." % len(rsp))
    
    # Create data generator for batch predictions
    generator_batch_size = params["generator_batch_size"]
    test_steps = int(np.ceil(len(rsp) / generator_batch_size))
    
    test_gen = data_merge_generator(
        rsp, ge, md, generator_batch_size, 
        params, merge_preserve_order=True, verbose=True
    )
  
    # Perform batch predictions
    try:
        test_pred, test_true = batch_predict(model, test_gen, test_steps)
    except ValueError as e:
        print("ValueError in batch_predict(): \n" + str(e),
              flush=True)
        output_dir = params["output_dir"]
        with open(output_dir + "/test-empty.txt", "w") as fp:
            fp.write("EMPTY\n")
        print("do_infer(): EMPTY.", flush=True)

        # Dump stack:
        info = sys.exc_info()
        s = traceback.format_tb(info[2])
        sys.stdout.write('\n\nEXCEPTION in batch_predict(): \n' +
                         repr(e) + ' ... \n' + ''.join(s))
        sys.stdout.write('\n')
        sys.stdout.flush()
        return

    # ------------------------------------------------------
    # Save raw predictions to a dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true,
        y_pred=test_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # Compute and save performance scores (optional)
    # ------------------------------------------------------
    if True: # params.get("calc_infer_scores", False):
        print("do_infer(): output_dir: " + params["output_dir"])
        try:
            test_scores = frm.compute_performance_scores(
                y_true=test_true, 
                y_pred=test_pred, 
                stage="test",
                metric_type=params["metric_type"],
                output_dir=params["output_dir"]
            )
        except ValueError as e:
            print("ValueError in compute_performance_scores(): \n" + str(e),
                  flush=True)
            output_dir = params["output_dir"]
            with open(output_dir + "/NaN.txt", "w") as fp:
                fp.write("NaN\n")
            print("do_infer(): NaN.", flush=True)
            return
            

    print("do_infer(): DONE.")
    sys.stdout.flush()

    return True


def main(args):
    """
    Main function to initialize parameters and run inference.

    Args:
        args (list): Command-line arguments.
    """
    # Combine parameter definitions from preprocessing, training, and inference stages
    additional_definitions = preprocess_params + train_params + infer_params

    # Initialize inference configuration
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_default_model.txt",
        additional_definitions=additional_definitions,
        required=None
    )

    # Run model inference
    test_scores = run(params)

    # Record inference duration
    infer_end_time = time.time()
    print_duration("Infering", infer_start_time, infer_end_time)
    print("\nFinished model inference.")


if __name__ == "__main__":
    # Record the start time for inference
    infer_start_time = time.time()
    main(sys.argv[1:])
