from Vector2D import Vector2D
import pandas as pd
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
            upper_limit = upper_lim_coord[1]
            lower_limit = lower_lim_coord[1]
        elif limit_orr == 'ver':
            upper_limit = upper_lim_coord[0]
            lower_limit = lower_lim_coord[0]
        else:
            msg = 'The limit orientation is either \'hor\' (horizontal),' \
                  'or \'ver\' (vertical); not {}'.format(limit_orr)
            raise AttributeError(msg)

        return upper_limit, lower_limit

    def __repr__(self):
        return '{} with {}'.format(__class__.__name__, self.csv_file)

    def position_preference(self, plot=False):
        total_frames = self.shape[0]-1
        nose_y = self.df['nose'].y.values.tolist()
        left_ear_y = self.df['left_ear'].y.values.tolist()
        right_ear_y = self.df['right_ear'].y.values.tolist()

        lower_environment = 0
        upper_environment = 0
        for i in range(total_frames):
            ear_test = (left_ear_y[i], right_ear_y[i])
            if ear_test > (self.lower_limit,)*2 \
                    and nose_y[i] > self.lower_limit:
                lower_environment += 1

            elif right_ear_y[i] > self.lower_limit \
                    and nose_y[i] > self.lower_limit:
                lower_environment += 1

            elif left_ear_y[i] < self.upper_limit \
                    and nose_y[i] < self.upper_limit:
                upper_environment += 1

            elif right_ear_y[i] < self.upper_limit \
                    and nose_y[i] < self.upper_limit:
                upper_environment += 1

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
        rows = self.shape[0]
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

    @property
    def df(self):
        def usecols_gen(total=self.shape[1]):
            for i in range(1, total+1, 3):
                yield [0]+[x for x in range(i, i+3)]

        if self.normalize:
            def y_normalizer(y_coord):
                if self.invert_y:
                    return 1 - y_coord / self.y_max
                else:
                    return y_coord / self.y_max

        result = {}
        for body_part, columns in zip(self.body_parts, usecols_gen()):
            body_part_df = self.read_csv(self.csv_file, usecols=columns)
            if self.normalize:
                body_part_df.x = body_part_df.x.apply(lambda x_coord:
                                                      x_coord / self.x_max)
                # Inspector warning irrelevant
                body_part_df.y = body_part_df.y.apply(y_normalizer)
            elif self.invert_y:
                body_part_df.y = body_part_df.y.apply(lambda y_coord:
                                                      self.y_max - y_coord)
            result[body_part] = body_part_df
        return result

    @property
    def df_interpolated(self):
        return

    @property
    def body_parts(self):
        """Instantiates a list with names of the body parts in the dataframe"""
        body_part_row = pd.read_csv(self.csv_file, index_col=0,
                                    skiprows=1, nrows=1)
        return body_part_row.columns.values.tolist()[::3]

    @property
    def shape(self):
        rows, columns = self.read_csv(self.csv_file).shape
        assert (columns / 3).is_integer(), 'Data file is invalid'
        return rows, columns

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

    @staticmethod
    def read_csv(csv_file, usecols=None):
        return pd.read_csv(csv_file, engine='c', delimiter=',', index_col=0,
                           skiprows=2, usecols=usecols, header=0)
