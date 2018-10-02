import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import cv2

from Vector2D import Vector2D
from collections import deque
from os.path import isfile, join, exists
from os import listdir

from DLC_analysis_settings import (DATA_FOLDER_NAME, MIN_LIKELIHOOD,
                                   MAX_DIF)


class DLCsv:
    def __init__(self, csv_filename, normalize=False, invert_y=False,
                 video_file=None, x_max=None, y_max=None, boarder_or=None,
                 upper_boarder=None, lower_boarder=None, lasso_num=None):
        """
        Python class to analyze csv files from DeepLabCut (DLC)

        Parameters
        ----------
        csv_filename: str
            Name of csv file to be analysed; with or without file-extension
        normalize: boolean, default False
            Normalizes the coordinates. Requires x_max and y_max to be defined
        invert_y: boolean, default False
            Inverts the y axis, yielding a traditional cartesian system;
            origin on the bottom left. Requires x_max and y_max to be defined
        video_file: {None, True, str}, default None
            Defines the name (str) or the existence of a video file
            in the local directory (True).

            None: No action

            str:  The video matching the string will be selected.
                  String must include file extension for the decoder.

            True: If there is only one video file, it will be selected.
                  If there are several, the user can submit the name of
                  the video file to be used during runtime.

        x_max: {None, int}, default None
            Maximum x value, can be extracted from video sample or defined
        y_max: {None, int}, default None
            Maximum y value, can be extracted from video sample or defined
        boarder_or: {None, 'hor', 'ver', 'lasso'}, default None
            Optional. A lower and an upper boarder can be defined.
            The boarders can be oriented both horizontally (hor)
            or vertically (ver). If vertical: lower -> right; upper -> left.

            With the use of the position_preference method, the ratio of time
            spent in the upper; the lower; the mid portion can be calculated.

            For boarder_or to function, video_file or lower_boarder and
            upper_boarder has to be defined.

            To use of lasso, video_file and lasso_num has to be defined.
        lower_boarder: int
            See boarder_or
        upper_boarder: int
            See boarder_or
        lasso_num: int
            Number of lasso selections. See boarder_or for more context
        """
        if not isinstance(csv_filename, str) and not csv_filename.endswith(
                '.csv'):
            msg = 'The argument has to be a string with the name of a csv file'
            raise AttributeError(msg)

        if (boarder_or != 'hor' and boarder_or != 'ver') \
                and boarder_or is not None:
            msg = 'The boarder orientation must be submitted ' \
                  'in string format,\n and is either \'hor\' (horizontal), ' \
                  'or \'ver\' (vertical); not {}'.format(boarder_or)
            raise AttributeError(msg)

        if not (isinstance(x_max, (int, float, type(None))) and
                isinstance(y_max, (int, float, type(None)))) and not (
                y_max.is_integer() and x_max.is_integer()):
            msg = 'x and y max are integers; not {}; {}'.format(x_max, y_max)
            raise AttributeError(msg)

        if normalize is True:
            msg = 'x max and y max has to defined in order to normalize'
            raise AttributeError(msg)

        elif invert_y is True:
            msg = 'y max has to defined in order to invert the y axis'
            raise AttributeError(msg)

        self.invert_y = invert_y
        self.normalize = normalize
        self.boarder_or = boarder_or
        self.lasso_num = lasso_num

        # Import the csv file
        self.csv_filename = csv_filename
        csv_path = os.path.join(DATA_FOLDER_NAME, csv_filename)

        type_dict = {'coords': int, 'x': float,
                     'y': float, 'likelihood': float}
        self.raw_df = pd.read_csv(csv_path, engine='c', delimiter=',',
                                  index_col=0, skiprows=1, header=[0, 1],
                                  dtype=type_dict, na_filter=False)

        self.n_rows, self.n_columns = self.raw_df.shape

        # Get name of body parts
        csv_multi_i = list(self.raw_df)
        body_parts = []
        for i in range(0, len(csv_multi_i), 3):
            body_parts.append(csv_multi_i[i][0])
        self.body_parts = body_parts

        self.video_file = video_file
        vid_test = video_file is True or isinstance(video_file, str)
        if vid_test:
            self.frame, self.x_max, self.y_max = self.get_video_data(
                filename=video_file)
        else:
            self.x_max = x_max
            self.y_max = y_max

        if boarder_or == 'hor' or boarder_or == 'ver':
            # 0: Use the x coordinate(s) as the border
            # 1: Use the y coordinate(s) as the border
            or_dic = {'ver': 0, 'hor': 1}

            if vid_test:
                plt.imshow(self.frame)
                plt.title('Lower limit')
                # coordinate for the lower boarder
                lower_var = plt.ginput()[0]
                plt.title('Upper limit')
                # coordinate for the upper boarder
                upper_var = plt.ginput()[0]

                lower_var = lower_var[or_dic[boarder_or]]
                upper_var = upper_var[or_dic[boarder_or]]

            elif isinstance(lower_boarder, int) \
                    and isinstance(upper_boarder, int):
                lower_var = lower_boarder
                upper_var = upper_boarder

            else:
                msg = 'Either video file, or lower and upper boarder' \
                      'has to be defined'
                raise AttributeError(msg)

            norm_dic = {'ver': self.x_max, 'hor': self.y_max}
            if self.invert_y:
                self.lower_boarder = y_max - lower_var
                self.upper_boarder = y_max - upper_var
                if normalize:
                    self.lower_boarder /= norm_dic[boarder_or]
                    self.upper_boarder /= norm_dic[boarder_or]

            elif normalize:
                self.lower_boarder = lower_var / norm_dic[boarder_or]
                self.upper_boarder = upper_var / norm_dic[boarder_or]

            else:
                self.lower_boarder = lower_var
                self.upper_boarder = upper_var

        elif boarder_or == 'lasso' and vid_test:
            x = 6
            x.isinteger

    def __repr__(self):
        header = '{}(\"{}\"):\n'.format(__class__.__name__, self.csv_filename)

        line_i = 'norm={}, inv_y={}, vid={},\n'.format(
            self.normalize, self.invert_y, self.video_file)

        line_ii = 'x_max={}, y_max={}'.format(self.x_max, self.y_max)

        base = header + line_i + line_ii

        if self.boarder_or is not None:
            line_iii = ',\nboarder_or=\"{}\"{}'.format(
                self.boarder_or,
                ', upper={}, lower={}'.format(self.upper_boarder,
                                              self.lower_boarder)
                if self.boarder_or != 'lasso' else
                ', lasso_num={}'.format(self.lasso_num)
            )
            return base + line_iii

        return base

    @staticmethod
    def get_video_data(filename, frame_loc='middle', path='.'):
        if isinstance(filename, str):
            user_video = os.path.join(DATA_FOLDER_NAME, filename)
        elif filename is True:
            videos = [v for v in listdir(path) if isfile(join(path, v)) and
                      v.endswith('.mp4')]
            if len(videos) == 1:
                user_video = videos[0]
            else:
                print('More than one video in directory')
                msg = 'Please type the filename of the video\n ' \
                      'that is to be used; example \'my_video.mp4\': '
                user_video = input(msg)
        else:
            msg = 'filename has to be defined as True, or video file name\n' \
                  'in string format.'
            raise AttributeError(msg)

        if user_video.endswith('.mp4') or user_video.endswith('.avi'):
            msg = 'Invalid video file.'
            raise TypeError(msg)

        cap = cv2.VideoCapture(user_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_loc == 'middle':
            target_frame = frame_count / 2
        else:
            msg = 'frame_loc can only be halfway; start; end,\n' \
                  'and not {}'.format(frame_loc)
            raise AttributeError(msg)

        cap.set(1, target_frame - 1)

        res, frame = cap.read()
        assert res, 'Could not extract frame from video'

        return frame, width, height

    def clean_df(self, min_like=MIN_LIKELIHOOD, max_dif=MAX_DIF, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        def bad_coords(comp):
            original = new_df.loc[:, (body_part, comp)].values
            minus_first = np.delete(original, 0, 0)
            minus_last = np.delete(original, -1, 0)

            ele_dif = np.subtract(minus_first, minus_last)
            bad_values = deque(np.greater_equal(ele_dif, max_dif))
            bad_values.appendleft(False)
            return bad_values

        new_df = self.raw_df.copy()

        for body_part in self.body_parts:
            """Clean low likelihood values"""

            new_df.loc[new_df[(body_part, 'likelihood')] < min_like,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

            """Clean high velocity values"""

            invalid_coords = np.logical_or(bad_coords('x'), bad_coords('y'))

            new_df.loc[invalid_coords,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

        if self.normalize:
            new_df.loc[:, (slice(None), 'x')] = \
                new_df.loc[:, (slice(None), 'x')] / self.x_max
            if self.invert_y:
                new_df.loc[:, (slice(None), 'y')] = \
                    (self.y_max - new_df.loc[:, (slice(None), 'y')]) \
                    / self.y_max
            else:
                new_df.loc[:, (slice(None), 'y')] = \
                    new_df.loc[:, (slice(None), 'y')] / self.y_max
        elif self.invert_y:
            new_df.loc[:, (slice(None), 'y')] = \
                self.y_max - new_df.loc[:, (slice(None), 'y')]

        if save is True:
            csv_name = 'cleaned_{}.csv'.format(self.csv_filename)
            new_df.to_csv(csv_name, sep='\t')

        return new_df

    def interpolated_df(self, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.clean_df()

        for body_part in self.body_parts:
            for comp in ('x', 'y'):
                new_df.loc[:, (body_part, comp)] = \
                    new_df.loc[:, (body_part, comp)].interpolate(
                        method='spline', order=4,
                        limit_area='inside')
        if save is True:
            csv_name = 'interpolated_{}.csv'.format(self.csv_filename)
            new_df.to_csv(csv_name, sep='\t')

        return new_df

    def position_preference(self, state='interpolated',
                            plot=False, boarder_or='hor'):
        if boarder_or == 'hor':
            or_var = 'y'
        elif boarder_or == 'ver':
            or_var = 'x'
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(boarder_or)
            raise AttributeError(msg)

        use_df = self.get_state(state)
        total_frames = self.n_rows - 1
        nose = use_df['nose'][or_var].values.tolist()
        left_ear = use_df['left_ear'][or_var].values.tolist()
        right_ear = use_df['right_ear'][or_var].values.tolist()

        if self.invert_y or boarder_or == 'ver':
            def nose_loc_test(index):
                return nose[index] < self.lower_boarder or \
                       nose[index] > self.upper_boarder

            def ear_loc_test(index, limit):
                if limit == 'upper':
                    return left_ear[index] < self.lower_boarder or \
                           right_ear[index] > self.lower_boarder
                elif limit == 'lower':
                    return left_ear[index] < self.upper_boarder or \
                           right_ear[index] > self.upper_boarder

        else:
            def nose_loc_test(index):
                return nose[index] > self.lower_boarder or \
                       nose[index] < self.upper_boarder

            def ear_loc_test(index, limit):
                if limit == 'lower':
                    return left_ear[index] > self.lower_boarder or \
                           right_ear[index] < self.lower_boarder
                elif limit == 'upper':
                    return left_ear[index] > self.upper_boarder or \
                           right_ear[index] < self.upper_boarder

        lower_environment = 0
        upper_environment = 0
        for i in range(total_frames):
            if nose_loc_test(i):
                if ear_loc_test(i, 'lower'):
                    lower_environment += 1
                elif ear_loc_test(i, 'upper'):
                    upper_environment += 1

        percent_lower = (lower_environment / total_frames) * 100
        percent_upper = (upper_environment / total_frames) * 100

        rest = 100 - percent_lower - percent_upper

        if plot:
            labels = ('Top', 'Bottom', 'Elsewhere')
            sizes = (percent_upper, percent_lower, rest)

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Ensures that pie is drawn as a circle.
            plt.show()
        else:
            return percent_lower, percent_upper

    def angle_df(self, body_part_centre, body_part_1, body_part_2):
        """Return a csv with the angle between three body parts per frame"""
        for i in range(self.n_rows):
            vector_centre = Vector2D(*self.bp_coords(
                body_part_centre, i))

            vector_body_part_1 = Vector2D(*self.bp_coords(
                body_part_1, i)) - vector_centre
            vector_body_part_2 = Vector2D(*self.bp_coords(
                body_part_2, i)) - vector_centre
            return vector_body_part_1 @ vector_body_part_2

    def bp_coords(self, body_part, row_index, state='interpolated'):
        """Returns body part coordinate from"""
        if not isinstance(body_part, str):
            msg = 'body_part has to be string'
            raise AttributeError(msg)
        if row_index % 1 != 0:
            msg = 'row_index must be an integer'
            raise AttributeError(msg)
        
        use_df = self.get_state(state)
        row = use_df[body_part].loc[row_index].tolist()
        return row[0], row[1]

    def view(self, state='raw'):
        """For viewing data frame in terminal; don't use in jupyter notebook"""
        use_df = self.get_state(state)
        for body_part in self.body_parts:
            print(use_df[body_part])

    def get_state(self, state):
        state_dict = {'raw': self.raw_df,
                      'cleaned': self.clean_df(),
                      'interpolated': self.interpolated_df()}
        return state_dict[state]
