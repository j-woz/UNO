""" Preprocessing of raw data to generate datasets for UNO Model. """
import time
preprocess_start_time = time.time()
import sys
import os
from pathlib import Path
from typing import Dict, List, Union
from uno_utils_improve import print_duration, get_common_samples, get_column_ranges, subset_data

# Script Dependencies: pandas, numpy, joblib, scikit-learn

# [Req] Import params
from params import app_preproc_params, model_preproc_params, app_train_params, model_train_params

import numpy as np
import pandas as pd
import joblib
import textwrap

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
)


filepath = Path(__file__).resolve().parent  # [Req]

# IMPROVE imports
from improve import framework as frm

# from improve import dataloader as dtl  # This is replaced with drug_resp_pred
from improve import (
    drug_resp_pred as drp,
)  # some funcs from dataloader.py were copied to drp


# ---------------------
# [Req] Parameter lists
# ---------------------
preprocess_params = app_preproc_params + model_preproc_params
train_params = app_train_params + model_train_params

# ------------------------------------------------------------


def scale_df(
    df: pd.DataFrame, scaler_name: str = "std", scaler=None, verbose: bool = False
):
    """Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        df: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = df.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "maxabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        elif scaler_name in ["l1", "l2", "max"]:
            scaler = Normalizer(norm=scaler_name)
        elif scaler_name == "power_yj":
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            print(
                f"The specified scaler ({scaler_name}) is not implemented (no df scaling)."
            )
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm

    # Remove rows with NaN or inf values and print proportion of rows removed
    rows_before = df.shape[0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df.dropna(inplace=True)  # Remove rows with NaN values
    rows_after = df.shape[0]
    proportion_removed = (rows_before - rows_after) / rows_before
    print(f"Proportion of rows removed for corrupted data: {proportion_removed:.3%}")

    # Return value
    return df, scaler


def run(params: Dict):
    """Execute data pre-processing for UNO model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    # Create output dir to save preprocessed data)
    frm.create_outdir(outdir=params["ml_data_outdir"])
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Reading hyperparameters
    # ------------------------------------------------------
    preprocess_debug = params["preprocess_debug"]
    preprocess_subset_data = params["preprocess_subset_data"]


    temp_start_time = time.time()
    # ------------------------------------------------------
    # [Req] Load omics data
    # ------------------------------------------------------
    print("\nLoading omics data.")
    omics_obj = drp.OmicsLoader(params)
    # print(omics_obj)
    gene_expression_file = params["x_data_canc_files"][0][0]
    print("Loading file: ", gene_expression_file)
    ge = omics_obj.dfs[gene_expression_file]
    ge["improve_sample_id"] = ge['improve_sample_id'].astype(str)  # To fix mixed dytpes error
    # Add ge prefix to identify gene expression columns later 
    # (all but the sample_id first column) (needed to choose block input sizes in train)
    first_column = ge.iloc[:, :1]
    rest_columns = ge.iloc[:, 1:].add_prefix('ge.')
    ge = pd.concat([first_column, rest_columns], axis=1)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Load drug data
    # ------------------------------------------------------
    print("\nLoading drugs data.")
    drugs_obj = drp.DrugsLoader(params)
    drug_data_file = params["x_data_drug_files"][0][0]
    print("Loading file: ", drug_data_file)
    md = drugs_obj.dfs[drug_data_file]  # return mordred drug descriptors
    md = md.reset_index()  # Needed to do scaling and merging as wanted
    md["improve_chem_id"] = md['improve_chem_id'].astype(str)  # To fix mixed dytpes error
    # ------------------------------------------------------

    temp_end_time = time.time()
    print("")
    print_duration("Loading Data", temp_start_time, temp_end_time)

    # Check loaded data if debug mode on
    if preprocess_debug:
        print("Loaded Gene Expression:")
        print(ge.head())
        print(ge.shape)
        print("")
        print("Loaded Mordred Descriptors:")
        print(md.head())
        print(md.shape)
        print("")

    # ------------------------------------------------------
    # To-Do: USE LINCS
    # ------------------------------------------------------
    # Gene selection (LINCS landmark genes)
    # if params["use_lincs"]:
    #     genes_fpath = filepath + "/raw_data/x_data/landmark_genes"
    #     ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])
    # ------------------------------------------------------

    temp_start_time = time.time()
    # ------------------------------------------------------
    # Data prep to create scaler on
    # ------------------------------------------------------
    # Load and combine train and val responses
    rsp_tr = drp.DrugResponseLoader(
        params, split_file=params["train_split_file"], verbose=False
    ).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(
        params, split_file=params["val_split_file"], verbose=False
    ).dfs["response.tsv"] # Note: Study column is integer but expects str giving warning
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)
    # Only keep relevant parts of response
    rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

    # Keep only common samples between canc, drug, and response
    ge_sub, md_sub, rsp_sub = get_common_samples(
        ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
    )

    # Check shapes if debug mode on
    if preprocess_debug:
        print(textwrap.dedent(f"""
            Gene Expression Shape Before Subsetting With Response: {ge.shape}
            Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
            Mordred Shape Before Subsetting With Response: {md.shape}
            Mordred Shape After Subsetting With Response: {md_sub.shape}
            Response Shape Before Merging With Data: {rsp.shape}
            Response Shape After Merging With Data: {rsp_sub.shape}
        """))


    # ------------------------------------------------------
    # Create Feature Scaler
    # ------------------------------------------------------
    # Note: these scale before merging, so doesn't overweight according
    # to common treatments/cell lines
    print("\nCreating Feature Scalers\n")

    # Create gene expression scaler
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["ge_scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # Create mordred descriptor scaler
    _, md_scaler = scale_df(md_sub, scaler_name=params["md_scaling"])
    md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    joblib.dump(md_scaler, md_scaler_fpath)
    print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub, md_sub   # Clean Up
    # ------------------------------------------------------
    temp_end_time = time.time()
    print_duration("Creating Scalers", temp_start_time, temp_end_time)


    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }

    for stage, split_file in stages.items():

        split_start_time = time.time()
        print(f"Stage: {stage.upper()}")
        # --------------------------------------------------
        # Data prep
        # --------------------------------------------------
        # Load split responses
        rsp = drp.DrugResponseLoader(params, split_file=split_file, verbose=False).dfs[
            "response.tsv"
        ]
        # Only keep relevant parts of response
        rsp = rsp[[params["canc_col_name"], params["drug_col_name"], params["y_col_name"]]]

        # Keep only common samples between canc, drug, and response
        ge_sub, md_sub, rsp_sub = get_common_samples(
            ge, md, rsp, params["canc_col_name"], params["drug_col_name"]
        )

        # Check shapes if debug mode on
        if preprocess_debug:
            print(textwrap.dedent(f"""
                Gene Expression Shape Before Subsetting With Response: {ge.shape}
                Gene Expression Shape After Subsetting With Response: {ge_sub.shape}
                Mordred Shape Before Subsetting With Response: {md.shape}
                Mordred Shape After Subsetting With Response: {md_sub.shape}
                Response Shape Before Merging With Data: {rsp.shape}
                Response Shape After Merging With Data: {rsp_sub.shape}
            """))

        # Scale features
        temp_start_time = time.time()
        print("\nScaling data")
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler)  # scale gene expression
        md_sc, _ = scale_df(md_sub, scaler=md_scaler)  # scale Mordred descriptors
        temp_end_time = time.time()
        print_duration(f"Applying Scaler to {stage.capitalize()}", temp_start_time, temp_end_time)
        if preprocess_debug:
            print("Gene Expression Scaled:")
            print(ge_sc.head())
            print(ge_sc.shape)
            print("")
            print("Mordred Descriptors Scaled:")
            print(md_sc.head())
            print(md_sc.shape)
            print("")

        # --------------------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # --------------------------------
            
        # Subset if setting true (for testing)
        if preprocess_subset_data:
            # Define the total number of samples and the proportions for each stage
            total_num_samples = 5000
            stage_proportions = {"train": 0.8, "val": 0.1, "test": 0.1}   # should represent proportions given
            # Shuffle with num_samples set by total and stage
            rsp_sub = subset_data(rsp_sub, stage, total_num_samples, stage_proportions)
            

        # Save final dataframe to the constructed file paths
        temp_start_time = time.time()
        print(f"Saving {stage.capitalize()} Data (unmerged) to Parquet")
        # [Req] Build data name
        data_fname = frm.build_ml_data_name(params, stage)
        ge_fname = f"ge_{data_fname}"
        md_fname = f"md_{data_fname}"
        rsp_fname = f"rsp_{data_fname}"
        # [Req] Save dataframes
        ge_sc.to_parquet(Path(params["ml_data_outdir"]) / ge_fname)
        md_sc.to_parquet(Path(params["ml_data_outdir"]) / md_fname)
        rsp_sub.to_parquet(Path(params["ml_data_outdir"]) / rsp_fname)
        # [Req] Save y dataframe for the current stage
        # Note that this requires the rsp df to have the same order after merging to be comparable
        ydf = rsp_sub[['improve_sample_id', 'improve_chem_id', params["y_col_name"]]]
        frm.save_stage_ydf(ydf, params, stage)
        temp_end_time = time.time()
        print_duration(f"Saving {stage.capitalize()} Dataframes", temp_start_time, temp_end_time)

        split_end_time = time.time()
        print_duration(f"Processing {stage.capitalize()} Data", split_start_time, split_end_time)

    # Print total duration
    preprocess_end_time = time.time()
    print_duration(
        f"Preprocessing Data (All)", preprocess_start_time, preprocess_end_time
    )

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params  # Required for HPO
    # original: additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="uno_default_model.txt",
        additional_definitions=additional_definitions,
        # required=req_preprocess_params,
        required=None,
    )
    ml_data_outdir = run(params)
    print(
        "\nFinished UNO pre-processing (transformed raw DRP data to model input ML data)."
    )


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
