#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# feature importance
python $SCRIPT_DIR/feature_importance.py  --model_type=single --model_path=sherlock_None.pt
python $SCRIPT_DIR/feature_importance.py  --model_type=single --model_path=all_None.pt --topic=num-directstr_thr-0_tn-400

python $SCRIPT_DIR/feature_importance.py  --model_type=CRF --model_path=CRF_pre.pt 
python $SCRIPT_DIR/feature_importance.py  --model_type=CRF --model_path=CRF+LDA_pre.pt --topic=num-directstr_thr-0_tn-400


cd $SCRIPT_DIR/../model
python train_CRF_LC.py -c params/crf_configs/CRF.txt --multi_col_only=true --comment=path 
python train_CRF_LC.py -c params/crf_configs/CRF+LDA.txt --multi_col_only=true --comment=pathL 

python train_CRF_LC.py -c params/crf_configs/CRF.txt  --comment=path 
python train_CRF_LC.py -c params/crf_configs/CRF+LDA.txt  --comment=pathL 

cd $SCRIPT_DIR
python per_type.py
