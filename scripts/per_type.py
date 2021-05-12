from analysis_functions import data_gen_all
import os

path = os.path.join(os.path.dirname(__file__), '..', 'results/CRF_log/type78/CRF_path/outputs')
pathL = os.path.join(os.path.dirname(__file__), '..', 'results/CRF_log/type78/CRF+LDA_pathL/outputs')

path_multi_col = os.path.join(os.path.dirname(__file__), '..', 'results/CRF_log/type78/CRF_path_multi-col/outputs')
pathL_multi_col = os.path.join(os.path.dirname(__file__), '..', 'results/CRF_log/type78/CRF+LDA_pathL_multi-col/outputs')


data_gen_all(path, pathL, 'all-tables', './output')
data_gen_all(path_multi_col, pathL_multi_col, 'multi-col', './output')

