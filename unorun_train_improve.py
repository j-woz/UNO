
from pathlib import Path
import time

from improvelib.applications.drug_response_prediction.config \
     import DRPTrainConfig

# Import custom utility functions
from uno_utils_improve import print_duration

from uno_train_improve import load_data, do_train
from uno_infer_improve import do_infer

from params import (app_preproc_params, model_preproc_params,
                    app_train_params, model_train_params)

preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

# Get the current file path
filepath = Path(__file__).resolve().parent


def main():
    print("unorun ...")
    train_start_time = time.time()
    """Main function to run the model training."""
    additional_definitions = preprocess_params + train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    train_end_time = time.time()
    print_duration("Total Training Time:", train_start_time, train_end_time)
    print("unorun: OK.")


def run(params):

    (tr_ge, tr_md, tr_rsp, num_ge_columns, num_md_columns) = \
        load_data(params, stage="train")

    (vl_ge, vl_md, vl_rsp, num_ge_columns, num_md_columns) = \
        load_data(params, stage="val")

    model, val_scores = do_train(params,
                                 tr_ge, tr_md, tr_rsp,
                                 vl_ge, vl_md, vl_rsp,
                                 num_ge_columns, num_md_columns)

    (test_ge, test_md, test_rsp, num_ge_columns, num_md_columns) = \
        load_data(params, stage="test")

    do_infer(params, model, test_ge, test_md, test_rsp)


def initialize_parameters():
    additional_definitions = preprocess_params + train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="uno_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    return params


if __name__ == "__main__": main()
