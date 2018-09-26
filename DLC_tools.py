from Vector2D import Vector2D
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_file, normalize=False, invert_y=False,
                 x_max=None, y_max=None):
        if not isinstance(csv_file, str) and not csv_file.endswith('.csv'):
            msg = 'The argument has to be a string with the name of a csv file'
            raise AttributeError(msg)

        self.csv_file = str(csv_file)
        self.x_max = x_max
        self.y_max = y_max

        if invert_y and not isinstance(y_max, (int, float)):
            msg = 'y max has to defined in order to invert the y axis'
            raise AttributeError(msg)
        self.invert_y = invert_y

        if normalize and not isinstance((x_max, y_max), (int, float)):
            msg = 'x max and y max has to defined in order to normalize'
            raise AttributeError(msg)
        self.normalize = normalize

        """Import the csv file"""

        type_dict = {'coords': int, 'x': float,
                    'y': float, 'likelihood': float}
        self.raw_df = pd.read_csv(self.csv_file, engine='c', delimiter=',',
                                  index_col=0, skiprows=1, header=[0, 1],
                                  dtype=type_dict, na_filter=False)

        self.nrow, self.ncolumn = self.raw_df.shape

    def get_limits(self, limit_orr='hor'):
        """Returns the location preference of the subject"""
        if not isinstance(limit_orr, str):
            msg = 'Limit orientation has to be declared using string'
            raise AttributeError(msg)

        frame = self.get_video_frame()
        plt.imshow(frame)

        plt.title('Upper limit')
        upper_lim_coord = plt.ginput()
        plt.title('Lower limit')
        lower_lim_coord = plt.ginput()

        if limit_orr == 'hor':
            orr_var = 1     # Use the y coordinate(s) as the border
            if self.invert_y:
                return self.y_max - upper_lim_coord[0][orr_var],\
                       self.y_max - lower_lim_coord[0][orr_var]
        elif limit_orr == 'ver':
            orr_var = 0     # Use the x coordinate(s) as the border
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(limit_orr)
            raise AttributeError(msg)

        return upper_lim_coord[0][orr_var], lower_lim_coord[0][orr_var]

    def __repr__(self):
        return '{} with {}'.format(__class__.__name__, self.csv_file)

    def position_preference(self, plot=False, limit_orr='hor'):
        if limit_orr == 'hor':
            orr_var = 'y'
        elif limit_orr == 'ver':
            orr_var = 'x'
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(limit_orr)
            raise AttributeError(msg)

        upper_limit, lower_limit = self.get_limits(limit_orr=limit_orr)

        total_frames = self.nrow - 1
        nose = self.df['nose'][orr_var].values.tolist()
        left_ear = self.df['left_ear'][orr_var].values.tolist()
        right_ear = self.df['right_ear'][orr_var].values.tolist()

        if self.invert_y or limit_orr == 'ver':
            def nose_loc_test(index):
                return nose[index] < upper_limit or nose[index] > lower_limit

            def ear_loc_test(index, limit):
                if limit == 'upper':
                    return left_ear[index] < upper_limit or \
                           right_ear[index] > upper_limit
                elif limit == 'lower':
                    return left_ear[index] < lower_limit or \
                           right_ear[index] > lower_limit
        else:
            def nose_loc_test(index):
                return nose[index] > upper_limit or nose[index] < lower_limit

            def ear_loc_test(index, limit):
                if limit == 'upper':
                    return left_ear[index] > upper_limit or \
                           right_ear[index] < upper_limit
                elif limit == 'lower':
                    return left_ear[index] > lower_limit or \
                           right_ear[index] < lower_limit

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
        rows = self.nrow
        for row_index in range(1, rows):
            vector_centre = Vector2D(*self.bp_coords(
                body_part_centre, row_index))
            vector_body_part_1 = Vector2D(*self.bp_coords(
                body_part_1, row_index))-vector_centre
            vector_body_part_2 = Vector2D(*self.bp_coords(
                body_part_2, row_index))-vector_centre
            return vector_body_part_1@vector_body_part_2

    def bp_coords(self, body_part, row_index):
        """Returns body part coordinate from"""
        row = self.df[body_part].loc[str(row_index)].tolist()
        return row[0], row[1]

    def view(self, state='raw'):
        state_dict = {'raw': self.raw_df,
                      'cleaned': self.cleaned_df,
                      'interpolated': self.interpolated_df}
        use_df = state_dict[state]
        for body_part in self.body_parts:
            print(use_df[body_part])

    @property
    def cleaned_df(self, like_thresh=0.90, dif_thresh=80, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.raw_df.copy()
        if save:
            outfile = self.save('cleaned')

        for body_part in self.body_parts:
            new_df.loc[new_df[(body_part, 'likelihood')] < like_thresh,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan
            # for comp in ('x', 'y'):
            #     pass

        if self.normalize:
            new_df.loc[:, (slice(None), 'x')] = \
                new_df.loc[:, (slice(None), 'x')] / self.x_max
            if self.invert_y:
                new_df.loc[:, (slice(None), 'y')] = \
                    (self.y_max - new_df.loc[:, (slice(None), 'y')])\
                    / self.y_max
            else:
                new_df.loc[:, (slice(None), 'y')] = \
                    new_df.loc[:, (slice(None), 'y')] / self.y_max
        elif self.invert_y:
            new_df.loc[:, (slice(None), 'y')] = \
                self.y_max - new_df.loc[:, (slice(None), 'y')]
        if save:
            outfile.close()

        return new_df

    @property
    def interpolated_df(self, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.cleaned_df.copy()
        if save:
            outfile = self.save('interpolated')

        for body_part in self.body_parts:
            for comp in ('x', 'y'):
                new_df.loc[:, (body_part, comp)] = \
                    new_df.loc[:, (body_part, comp)].interpolate(
                        method='spline', order=4,
                        limit_area='inside')
        if save:
            outfile.close()

        return new_df

    @property
    def body_parts(self):
        """Instantiates a list with names of the body parts in the dataframe"""
        csv_headers = list(self.raw_df)

        body_parts = []
        for i in range(0, len(csv_headers), 3):
            body_parts.append(csv_headers[i][0])
        return tuple(body_parts)

    @staticmethod
    def save(state, cust_name=None):
        if cust_name is not None:
            outfile = open(cust_name + '.csv', 'w')
        else:
            outfile = open(state + '_data.csv', 'w')
        return outfile

    @staticmethod
    def get_video_frame(frame_loc='halfway', path='.'):
        import cv2
        from os import listdir
        from os.path import isfile, join, exists

        videos = [v for v in listdir(path)
                  if isfile(join(path, v)) and v.endswith(video_format)]
        if len(videos) == 1:
            user_video = videos[0]
        else:
            print('More than one video in directory')
            user_video = \
                input('Please type the filename of the video\n'
                      'that is to be used;example \'my_video.mp4\': ')

        assert exists(user_video), 'Can\'t find video file in directory'
        cap = cv2.VideoCapture(user_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_loc == 'halfway':
            target_frame = frame_count / 2
        else:
            msg = 'frame_loc can only be halfway; start; end,\n' \
                  'and not {}'.format(frame_loc)
            raise AttributeError(msg)

        cap.set(1, target_frame - 1)

        res, frame = cap.read()
        assert res, 'Could not extract frame from video'

        return frame
