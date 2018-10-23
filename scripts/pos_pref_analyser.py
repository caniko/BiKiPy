import pandas as pd
import pickle
import sys
import os

from dlca.video_analysis import get_video_data
from dlca.analysis import DLCPos
from dlca.readers import DLCsv, csv_iterator
from settings import DATA_FOLDER_NAME, OUTPUT_FOLDER_NAME


"""
sys.argv[1] = <.csv filename>: {'<name>', 'all'}
    If provided <name> only the file with filename will be analysed;
    'all' will analyse all files and return a .csv file with results to the
        root/output folder.

sys.argv[2] = <border_var>: {'hor', 'ver', 'lasso'}
    The orientation and style of border:
        hor:    horizontal
        ver:    vertical
        lasso:  define border(s) using lasso tool

sys.argv[3] = <video filename>: {'<name>', 'one'}, default str: 'one'
    Name of video file located in DATA_FOLDER

sys.argv[4] = <kwarg>: optional: jup
    Keyword argument that calls a specific function:
        jup:    prepare data for use in jupyter notebook
        ...
        new functions are added by request
"""


if len(sys.argv) <= 2:
    msg = 'The user has to pass at least two arguments'
    raise ValueError(msg)

if not os.path.exists(DATA_FOLDER_NAME):
    msg = 'The path to data folder is invalid'
    raise ValueError(msg)

if not sys.argv[1].endswith('.csv') and sys.argv[1] != 'all':
    msg = 'The file is not a .csv file'
    raise ValueError(msg)


if sys.argv[2] == 'hor' or sys.argv[2] == 'ver' or sys.argv[2] == 'lasso':
    border_or = sys.argv[2]
else:
    msg = 'The border orientation must be submitted ' \
          'in string format,\n and is either \'hor\' (horizontal), ' \
          '\'ver\' (vertical), or \'lasso\'; not {}'.format(sys.argv[2])
    raise AttributeError(msg)


if len(sys.argv) == 3 or len(sys.argv) == 4:
    if len(sys.argv) == 3 or sys.argv[3] == 'one':
        frame, x_max, y_max = get_video_data(True, path=DATA_FOLDER_NAME)

    elif len(sys.argv) == 4:
        if isinstance(sys.argv[3], str):
            frame, x_max, y_max = get_video_data(sys.argv[3],
                                                 path=DATA_FOLDER_NAME)
        else:
            msg = 'video_file is invalid'
            raise ValueError(msg)

    if sys.argv[1] != 'all':
        csv_file = os.path.join(DATA_FOLDER_NAME, sys.argv[1])
        df_obj = DLCsv(csv_file, x_max=x_max, y_max=y_max)
        pos_obj = DLCPos(df_obj.interpolate(), border_or, frame=frame)

        pos_obj.position_preference(plot=True)
    else:
        result = csv_iterator('position_preference', analysis_initi=DLCPos,
                              state='interpolated', path=DATA_FOLDER_NAME,
                              kwargs_for_initi={'border_or': border_or,
                                                'frame': frame})
        df = pd.DataFrame.from_dict(result, orient='index',
                                    columns=['Bottom', 'Top', 'Elsewhere'])
        df.index.name = 'Experiment'

        if not os.path.exists(OUTPUT_FOLDER_NAME):
            os.mkdir(OUTPUT_FOLDER_NAME)

        data_id = int(input('ID number for output data: '))
        out_csv_name = 'position_preference_{}.csv'.format(data_id)
        csv_path = os.path.join(OUTPUT_FOLDER_NAME, out_csv_name)

        df.to_csv(csv_path)

if len(sys.argv) == 5:
    if sys.argv[4] == 'jup':
        outfile = open('jup_prep.pckl', 'wb')
        pickle.dump(x_max, outfile)
        pickle.dump(y_max, outfile)
        
        raw_lower_boarder, raw_upper_boarder = \
            DLCPos.get_border(border_or, frame=frame)

        pickle.dump(raw_lower_boarder, outfile)
        pickle.dump(raw_upper_boarder, outfile)

        if border_or == 'lasso':
            lasso_num = input('Please define number of selections to be made '
                              '(use integer): ')
            if not isinstance(lasso_num, int):
                msg = 'Number of selections must be in integers'
                raise ValueError(msg)

        outfile.close()
        print('Data has been saved to data folder,\n'
              'and is ready for use in jupyter notebook')

    else:
        msg = 'Invalid argument passed to index 4; {}'.format(sys.argv[4])
        raise ValueError(msg)
