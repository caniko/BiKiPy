import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from dlca.video_analysis import get_video_data


class DLCPos:
    def __init__(self, pandas_df, border_or, normalize=False,
                 usr_lower=None, usr_upper=None, lasso_num=None,
                 video_file=None, frame=None, x_max=None, y_max=None,
                 notebook=False):
        """
        pandas_df: pandas.DataFrame object from DLCsv
            Data container with data to be analysed.
        border_or: {'hor', 'ver', 'lasso'}, default None
            Optional. A lower and an upper border can be defined.
            The borders can be oriented both horizontally (hor)
            or vertically (ver). If vertical: lower -> right; upper -> left.

            With the use of the position_preference method, the ratio of time
            spent in the upper; the lower; the mid portion can be calculated.

            For border_or to function, video_file or lower_border and
            upper_border has to be defined.

            To use of lasso, video_file and lasso_num has to be defined.
        normalize: bool, default False
            Define if pandas_df is normalized
        lower_border: int
            Variable to define lower border manually.
            See border_or for context
        upper_border: int
            Variable to define upper border manually.
            See border_or for context
        lasso_num: int
            Number of lasso selections. See border_or for more context
        video_file: {None, str}, default None
            Optional. Defines the name of the video file in local directory.
            Used to acquire resolution information, and a sample frame.

            Required if border_or == 'lasso' and frame == None.
            None: No action

            str:  The video file matching the string will be selected.
                  File extension must be included.

            True: If there is only one video file, it will be selected.
        frame: {str, numpy.array, None}, default None

        notebook: bool, default False
            Import user defined parameters from jup_data.txt for use in ipynb
        """
        if not isinstance(pandas_df, pd.DataFrame):
            msg = 'pandas_df has to be a pandas data frame'
            raise AttributeError(msg)

        if border_or != 'hor' and border_or != 'ver' and border_or != 'lasso':
            msg = 'The border orientation must be submitted ' \
                  'in string format,\n and is either \'hor\' (horizontal), ' \
                  '\'ver\' (vertical), or \'lasso\'; not {}'.format(border_or)
            raise AttributeError(msg)

        if border_or == 'lasso' and not isinstance(video_file, str) \
                and not isinstance(frame, str):
            msg = 'A frame is required to use the lasso tool ' \
                  'for defining borders.\n' \
                  'Frames can be acquired from image or video'
            raise AttributeError(msg)

        if video_file is not True and video_file is not None and (
                not isinstance(video_file, str)):
            msg = 'video_file must be defined as either True or string.'
            raise AttributeError(msg)

        if isinstance(lasso_num, int) and border_or != 'lasso':
            msg = 'lasso_num is to remain as None if border_or != \'lasso\''
            raise AttributeError(msg)
        else:
            self.lasso_num = lasso_num

        if not isinstance(normalize, bool):
            msg = 'normalize has to boolean'
            raise AttributeError(msg)

        if notebook is False:
            if isinstance(video_file, str) or video_file is True:
                frame, x_max, y_max = get_video_data(video_file)

            elif frame is not None:
                if isinstance(frame, type(np.zeros(0))):
                    pass
                elif isinstance(frame, str):
                    from matplotlib.image import imread

                    frame = imread(frame)
                else:
                    msg = 'The variable frame has to be a numpy array'
                    raise AttributeError(msg)

            if border_or == 'ver' or border_or == 'hor':
                lower_var, upper_var = \
                    self.get_border(border_or, frame=frame,
                                    usr_lower=usr_lower,
                                    usr_upper=usr_upper)

            elif border_or == 'lasso':
                areas = self.get_border(border_or, frame=frame,
                                        lasso_num=lasso_num)

        elif notebook is True:
            import pickle

            with open(os.path.normpath('jup_prep.pckl'), 'rb') as infile:
                self.x_max = pickle.load(infile)
                self.y_max = pickle.load(infile)

                if border_or == 'hor' or border_or == 'ver':
                    lower_var = pickle.load(infile)
                    upper_var = pickle.load(infile)
                elif border_or == 'lasso':
                    pass

        else:
            msg = 'notebook has to be a bool'
            raise AttributeError(msg)

        if border_or == 'hor' or border_or == 'ver':
            if normalize is True:
                norm_ref_dic = {'ver': x_max, 'hor': y_max}
                self.lower_border /= norm_ref_dic[border_or]
                self.upper_border /= norm_ref_dic[border_or]
            else:
                self.lower_border = lower_var
                self.upper_border = upper_var

        self.normalize = normalize
        self.pandas_df = pandas_df
        self.border_or = border_or

    def __repr__(self):
        header = '{}(\"{}\"):\n'.format(__class__.__name__, self.pandas_df)
        line_i = ',\nborder_or=\"{}\"{}'.format(
            self.border_or,
            ', upper={}, lower={}'.format(self.upper_border,
                                          self.lower_border)
            if self.border_or != 'lasso' else
            ', lasso_num={}'.format(self.lasso_num)
        )

        return header + line_i

    @staticmethod
    def get_border(border_or, frame=None, usr_lower=None, usr_upper=None,
                    lasso_num=None):
        if border_or == 'hor' or border_or == 'ver':
            # 0: Use the x coordinate(s) as the border
            # 1: Use the y coordinate(s) as the border
            or_dic = {'ver': 0, 'hor': 1}

            if frame is not None:
                plt.imshow(frame)
                plt.title('Lower limit')
                # coordinate for the lower border
                lower_var = plt.ginput()[0]
                plt.title('Upper limit')
                # coordinate for the upper border
                upper_var = plt.ginput()[0]

                lower_var = lower_var[or_dic[border_or]]
                upper_var = upper_var[or_dic[border_or]]

            elif isinstance(usr_lower, int) \
                    and isinstance(usr_upper, int):
                lower_var = usr_lower
                upper_var = usr_upper

            else:
                msg = 'Either video file, frame, or lower and upper border\n' \
                      'has to be defined'
                raise AttributeError(msg)

            return lower_var, upper_var

        elif border_or == 'lasso':
            return

    def position_preference(self, plot=False, border_or='hor'):
        if border_or == 'hor':
            or_var = 'y'
        elif border_or == 'ver':
            or_var = 'x'
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(border_or)
            raise AttributeError(msg)

        use_df = self.pandas_df
        total_frames = use_df.shape[0] - 1
        nose = use_df.nose[or_var].values
        left_ear = use_df.left_ear[or_var].values
        right_ear = use_df.right_ear[or_var].values

        # Disregard warnings as they arise from NaN being compared to numbers
        np.warnings.filterwarnings('ignore')

        if border_or == 'ver':
            nose_test = np.logical_or(np.less(nose, self.lower_border),
                                      np.greater(nose, self.upper_border))

            lower_test = np.logical_or(np.less(left_ear, self.lower_border,
                                               where=nose_test),
                                       np.less(right_ear, self.lower_border,
                                               where=nose_test),
                                       where=nose_test)

            lower_result = np.sum(np.extract(lower_test, nose_test))

            upper_test = np.logical_or(np.greater(left_ear, self.upper_border,
                                                  where=nose_test),
                                       np.greater(right_ear, self.upper_border,
                                                  where=nose_test),
                                       where=nose_test)

            upper_result = np.sum(np.extract(upper_test, nose_test))

        else:
            nose_test = np.logical_or(np.greater(nose, self.lower_border),
                                      np.less(nose, self.upper_border))

            lower_test = np.logical_or(np.greater(left_ear, self.lower_border,
                                                  where=nose_test),
                                       np.greater(right_ear, self.lower_border,
                                                  where=nose_test),
                                       where=nose_test)

            lower_result = np.sum(np.extract(lower_test, nose_test))

            upper_test = np.logical_or(np.less(left_ear, self.upper_border,
                                               where=nose_test),
                                       np.less(right_ear, self.upper_border,
                                               where=nose_test),
                                       where=nose_test)

            upper_result = np.sum(np.extract(upper_test, nose_test))

        percent_lower = (lower_result / total_frames) * 100
        percent_upper = (upper_result / total_frames) * 100

        rest = 100 - percent_lower - percent_upper

        if plot:
            print('Percent bottom: {:.2f}%\n'
                  'Percent top:    {:.2f}%'.format(percent_lower,
                                                   percent_upper))

            labels = ('Bottom', 'Top', 'Elsewhere')
            sizes = (percent_lower, percent_upper, rest)

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Ensures that pie is drawn as a circle.
            plt.show()
        else:
            return round(percent_lower, 2), round(percent_upper, 2), \
                   round(rest, 2)
