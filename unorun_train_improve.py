
from uno_train_improve import load_data
from uno_train_improve import do_train
from uno_infer_improve import do_infer

def run(params):

    
    
    (tr_ge, tr_md, tr_rsp, num_ge_columns, num_md_columns) = \
        load_data(params, stage="train")
         
    (vl_ge, vl_md, vl_rsp, num_ge_columns, num_md_columns) = \
        load_data(params, stage="val")

    model, val_scores = do_train(params, 
             tr_ge, tr_md, tr_rsp, 
             vl_ge, vl_md, vl_rsp,
             num_ge_columns, num_md_columns)

    do_infer(params, model, data)

    

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
