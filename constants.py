import ast
import multiprocessing
import os

from ConfigParser import SafeConfigParser

# Parse config file
parser = SafeConfigParser()
parser.read('config_CMS.txt')

if os.name == 'nt':
    prj_dir = parser.get('PATHS', 'win_prjdir')
elif os.name == 'mac' or os.name == 'posix':
    prj_dir = parser.get('PATHS', 'mac_prjdir')
else:
    print('Unsupported OS')

###############################################################################
# User modifiable values
#
#
###############################################################################
TAG = parser.get('PROJECT', 'TAG')
name_inp_fl = parser.get('PROJECT', 'name_inp_fl')
fixed_vars = ast.literal_eval(parser.get('PROJECT', 'fixed_vars'))

# Constants
DPI = 150  # dots per inch for saved figures
ncpu = multiprocessing.cpu_count() - 2  # Subtract 1 or more so it is free for other tasks

# Create directories
import utils
base_dir = prj_dir + os.sep + parser.get('PATHS', 'base_dir') + os.sep  # Input data directory
out_dir = prj_dir + os.sep + parser.get('PATHS', 'out_dir') + os.sep + parser.get('PROJECT', 'project_name') + os.sep
log_dir = out_dir + os.sep + 'Logs'
utils.make_dir_if_missing(base_dir)
utils.make_dir_if_missing(out_dir)
utils.make_dir_if_missing(log_dir)

# Machine learning specific variables
ML_TARGETS = parser.get('FORECAST', 'ML_TARGETS')
DO_CLASSIFICATION = parser.getboolean('FORECAST', 'DO_CLASSIFICATION')
model_pickle = parser.get('FORECAST', 'model_pickle')
use_disk_csv = parser.getboolean('FORECAST', 'USE_CSV_ON_DISK')
FILL_ROWS_W_MEDIAN = parser.getboolean('FORECAST', 'FILL_ROWS_W_MEDIAN')
compute_wtd_stats = parser.getboolean('FORECAST', 'compute_wtd_stats')
plot_model_importance = parser.getboolean('FORECAST', 'plot_model_importance')
