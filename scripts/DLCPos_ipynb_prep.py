from dlca.video_analysis import get_video_data
from dlca.analysis import DLCPos
from settings import DATA_FOLDER_NAME, BASE_FOLDER_NAME
import pickle
import sys
import os


"""Optional analysis preparation for jupyter notebook
Jupyter does not support matplotlib interaction yet.
Some functions must therefore be run before running DLCsv

sys.argv[1] = border_var
sys.argv[2] = video file name
"""

if len(sys.argv) != 3:
    msg = 'There has to be two system arguments to run script:\n' \
          'sys.argv[1] = border_var\n' \
          'sys.argv[2] = video file name'
    raise ValueError(msg)

try:
    os.chdir(DATA_FOLDER_NAME)
except OSError:
    msg = 'DATA_FOLDER_NAME has invalid path'
    raise AttributeError(msg)

f = open('jup_prep.pckl', 'wb')
video = sys.argv[2]
if not video.isdigit() and (isinstance(video, str)):
    frame, x_max, y_max = get_video_data(video)

    pickle.dump(x_max, f)
    pickle.dump(y_max, f)
else:
    msg = 'Video file is invalid'
    raise ValueError(msg)

if sys.argv[1] == 'hor' or sys.argv[1] == 'ver' or sys.argv[1] == 'lasso':
    border_or = sys.argv[1]
    raw_lower_boarder, raw_upper_boarder = \
        DLCPos.get_border(border_or, frame=frame)

    pickle.dump(raw_lower_boarder, f)
    pickle.dump(raw_upper_boarder, f)
else:
    msg = 'The border orientation must be submitted ' \
          'in string format,\n and is either \'hor\' (horizontal), ' \
          '\'ver\' (vertical), or \'lasso\'; not {}'.format(sys.argv[1])
    raise AttributeError(msg)

if border_or == 'lasso':
    lasso_num = input('Please define number of selections to be made '
                      '(use integer): ')
    if not isinstance(lasso_num, int):
        msg = 'Number of selections must be in integers'
        raise ValueError(msg)

f.close()
