import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import re
import cv2

from Vector2D import Vector2D
from collections import deque
from os.path import isfile, join, exists
from os import listdir

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_filename, normalize=False, invert_y=False,
                 video_file=None, x_max=None, y_max=None, boarder_orr=None,
                 upper_boarder=None, lower_boarder=None):
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
        boarder_orr: {None, 'hor', 'ver'}, default None
            Optionally, a lower and an upper boarder can be defined.
            The boarders can be oriented both horizontally (hor)
            or vertically (ver). If vertical: lower -> right; upper -> left.

            With the use of the position_preference method, the ratio of time
            spent in the upper; the lower; the mid portion can be calculated.

            For boarder_orr to function, video_file or
            upper_boarder and lower_boarder has to be defined.
            In order to either select boarder coordinates during
            runtime (video file), or define pre-defined values.
        upper_boarder: int
            See boarder_orr
        lower_boarder: int
            See boarder_orr
        """
        if not isinstance(csv_filename, str) and not csv_filename.endswith(
                '.csv'):
            msg = 'The argument has to be a string with the name of a csv file'
            raise AttributeError(msg)

        if boarder_orr == 'hor' or boarder_orr == 'ver' or \
                boarder_orr is not None:
            msg = 'The boarder orientation must be submitted' \
                  'in string format,\n and is either \'hor\' (horizontal), ' \
                  'or \'ver\' (vertical); not {}'.format(boarder_orr)
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

        # Import the csv file
        type_dict = {'coords': int, 'x': float,
                     'y': float, 'likelihood': float}
        self.raw_df = pd.read_csv(csv_filename, engine='c', delimiter=',',
                                  index_col=0, skiprows=1, header=[0, 1],
                                  dtype=type_dict, na_filter=False)
        self.csv_filename = csv_filename
        self.row_n, self.column_n = self.raw_df.shape

        # Get name of body parts
        csv_multi_i = list(self.raw_df)
        body_parts = []
        for i in range(0, len(csv_multi_i), 3):
            body_parts.append(csv_multi_i[i][0])
        self.body_parts = body_parts

        if video_file is True:
            frame, self.x_max, self.y_max = self.get_video_data(
                filename=video_file
            )
            self.video_file = video_file
        else:
            self.x_max = x_max
            self.y_max = y_max

        if boarder_orr is True:
            if video_file is True:
                plt.imshow(frame)

                plt.title('Upper limit')
                upper_var = plt.ginput()[0]  # coordinate for the upper boarder
                plt.title('Lower limit')
                lower_var = plt.ginput()[0]  # coordinate for the lower boarder

            elif upper_boarder is int and lower_boarder is int:
                upper_var = upper_boarder
                lower_var = lower_boarder

            else:
                msg = 'Either video file, or lower and upper boarder' \
                      'has to be defined'
                raise AttributeError(msg)

            if boarder_orr == 'hor':
                orr_var = 1  # Use the y coordinate(s) as the border
            elif boarder_orr == 'ver':
                orr_var = 0  # Use the x coordinate(s) as the border

            norm_dic = {0: self.x_max, 1: self.y_max}
            if self.invert_y:
                self.upper_boarder = y_max - upper_var[orr_var]
                self.lower_boarder = y_max - lower_var[orr_var]
                if normalize:
                    self.upper_boarder /= norm_dic[orr_var]
                    self.upper_boarder /= norm_dic[orr_var]

            elif normalize:
                self.upper_boarder = upper_var[orr_var] \
                                     / norm_dic[orr_var]
                self.lower_boarder = lower_var[orr_var] \
                                     / norm_dic[orr_var]

            else:
                self.upper_boarder = upper_var[orr_var]
                self.lower_boarder = lower_var[orr_var]

    def __repr__(self):
        return '{}({}):   norm={}; inv_y={}; vid={};\n'.format(
            __class__.__name__, self.csv_filename, self.normalize,
            self.invert_y, self.video_file), \
               'res_info: x_max={}; y_max={};\n'.format(
                   self.x_max, self.y_max), \
               'boarder:  upper={}; lower={}'.format(
                   self.upper_boarder, self.lower_boarder)

    @staticmethod
    def get_video_data(filename, frame_loc='middle', path='.'):
        if re.search(r'\.((avi)|(mp4))$', filename):
            user_video = filename

        elif filename is True:
            videos = [v for v in listdir(path)
                      if isfile(join(path, v)) and
                      re.search(r'\.((avi)|(mp4))$', v)]
            if len(videos) == 1:
                user_video = videos[0]
            else:
                print('More than one video in directory')
                msg = 'Please type the filename of the video\n ' \
                      'that is to be used;example \'my_video.mp4\': '
                user_video = input(msg)
                if not re.search(r'\.((avi)|(mp4))$', user_video):
                    msg = 'Invalid video file'
                    raise TypeError(msg)
        assert exists(user_video), 'Can\'t find video file in directory'

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

    @property
    def cleaned_df(self, like_thresh=0.90, dif_thresh=50, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        def bad_coords(comp):
            original = new_df.loc[:, (body_part, comp)].values
            minus_first = np.delete(original, 0, 0)
            minus_last = np.delete(original, -1, 0)

            ele_dif = np.subtract(minus_first, minus_last)
            bad_values = deque(np.greater_equal(ele_dif, dif_thresh))
            bad_values.appendleft(False)
            return bad_values

        new_df = self.raw_df.copy()

        for body_part in self.body_parts:
            """Clean low likelihood values"""

            new_df.loc[new_df[(body_part, 'likelihood')] < like_thresh,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

            """Clean high velocity values"""

            invalid_coords = np.logical_or(bad_coords('x'), bad_coords('y'))

            print(np.atleast_1d(invalid_coords))

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
            new_df.to_csv('cleaned_data.csv', sep='\t')

        return new_df

    @property
    def interpolated_df(self, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.cleaned_df.copy()

        for body_part in self.body_parts:
            for comp in ('x', 'y'):
                new_df.loc[:, (body_part, comp)] = \
                    new_df.loc[:, (body_part, comp)].interpolate(
                        method='spline', order=4,
                        limit_area='inside')
        if save is True:
            new_df.to_csv('interpolated_data.csv', sep='\t')

        return new_df

    def position_preference(self, state='interpolated',
                            plot=False, boarder_orr='hor'):
        if boarder_orr == 'hor':
            orr_var = 'y'
        elif boarder_orr == 'ver':
            orr_var = 'x'
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(boarder_orr)
            raise AttributeError(msg)

        use_df = self.get_state(state)
        total_frames = self.row_n - 1
        nose = use_df['nose'][orr_var].values.tolist()
        left_ear = use_df['left_ear'][orr_var].values.tolist()
        right_ear = use_df['right_ear'][orr_var].values.tolist()

        if self.invert_y or boarder_orr == 'ver':
            def nose_loc_test(index):
                return nose[index] < self.upper_boarder or \
                       nose[index] > self.lower_boarder

            def ear_loc_test(index, limit):
                if limit == 'upper':
                    return left_ear[index] < self.upper_boarder or \
                           right_ear[index] > self.upper_boarder
                elif limit == 'lower':
                    return left_ear[index] < self.lower_boarder or \
                           right_ear[index] > self.lower_boarder
        else:
            def nose_loc_test(index):
                return nose[index] > self.upper_boarder or \
                       nose[index] < self.lower_boarder

            def ear_loc_test(index, limit):
                if limit == 'upper':
                    return left_ear[index] > self.upper_boarder or \
                           right_ear[index] < self.upper_boarder
                elif limit == 'lower':
                    return left_ear[index] > self.lower_boarder or \
                           right_ear[index] < self.lower_boarder

        lower_environment = 0
        upper_environment = 0
        for i in range(total_frames):
            if nose_loc_test(i):
                if ear_loc_test(i, 'upper'):
                    upper_environment += 1
                elif ear_loc_test(i, 'lower'):
                    lower_environment += 1

        percent_upper = (upper_environment / total_frames) * 100
        percent_lower = (lower_environment / total_frames) * 100
        rest = 100 - sum(percent_lower, percent_upper)

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
        """Return a csv with the angle between three body parts per frame
        WIP
        """
        rows = self.row_n
        for row_index in range(1, rows):
            vector_centre = Vector2D(*self.bp_coords(
                body_part_centre, row_index))
            vector_body_part_1 = Vector2D(*self.bp_coords(
                body_part_1, row_index)) - vector_centre
            vector_body_part_2 = Vector2D(*self.bp_coords(
                body_part_2, row_index)) - vector_centre
            return vector_body_part_1 @ vector_body_part_2

    def bp_coords(self, body_part, row_index, state='interpolated'):
        """Returns body part coordinate from"""
        use_df = self.get_state(state)
        row = use_df[body_part].loc[str(row_index)].tolist()
        return row[0], row[1]

    def view(self, state='raw'):
        use_df = self.get_state(state)
        for body_part in self.body_parts:
            print(use_df[body_part])

    def get_state(self, state):
        state_dict = {'raw': self.raw_df,
                      'cleaned': self.cleaned_df,
                      'interpolated': self.interpolated_df}
        return state_dict[state]
