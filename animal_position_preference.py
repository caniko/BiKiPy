#from Vector2D import Vector2D


class DLC_position_preference:
    """ Used for calculating the percent-wise position preference of the animal.

        It is tested whether the animal prefers to be in the upper part, or
        lower part of the environment; 2 cm from each wall.

        In two_cm_upper_limit and two_cm_lower_limit it finds
        horizontal line 2 cm from the bottom line and 2 cm from the upper line.
        A prerequisites are:
        - video needs to be cropped in a certain way
        - origo is in the upper left corner """

    def __init__(self, total_y_pixel, left_ear, right_ear, nose):
        self.total_y_pixel = total_y_pixel    # example: for 388x188, total_y_pixel = 186
        self.left_ear = left_ear  # list of y coordinates for body part nr. 1
        self.right_ear = right_ear  # list of y coordinates for body part nr. 2
        self.nose = nose  # list of y coordinates for body part nr. 3l
    
    @property
    def two_cm_upper_limit(self):
        return self.total_y_pixel / 4.65

    @property
    def two_cm_lower_limit(self):
        return self.total_y_pixel / 1.18

    def position_preference(self):
        lower_environment = 0
        upper_environment = 0
        total_nr_of_frames = 0

        for i in range(len(self.nose)):
            if self.left_ear[i] > self.two_cm_lower_limit and self.nose[i] > self.two_cm_lower_limit:
                lower_environment += 1
            elif self.right_ear[i] > self.two_cm_lower_limit and self.nose[i] > self.two_cm_lower_limit:
                lower_environment += 1
            elif self.left_ear[i] < self.two_cm_upper_limit and self.nose[i] < self.two_cm_upper_limit:
                upper_environment += 1
            elif self.right_ear[i] < self.two_cm_upper_limit and self.nose[i] < self.two_cm_upper_limit:
                upper_environment += 1
            total_nr_of_frames += 1

        percent_upper = (upper_environment / total_nr_of_frames) * 100
        percent_lower = (lower_environment / total_nr_of_frames) * 100
        return ("Lower environment: {}% \nUpper environment: {}%".format(round(
            percent_lower, 2), round(percent_upper, 2)))


# example before incorporating with the DLCsv
if __name__ == '__main__':
    nose_y = [158, 74.8017916, 72.76470613, 71.83685446, 72.06032467, 77.660725, 22, 22, 22, 22, 88.91118813, 95.83561897, 98.05349469, 97.35266972, 99.20710927, 98.69586754, 99.39605695, 96.12210178, 98.44720972, 98.35878849, 104.328619, 104.5202456, 104.4529443]
    left_ear_y = [158, 85.095456, 80.73817682, 78.80043721, 78.64884067, 81.50397062, 22, 22, 22, 22, 86.60903049, 89.96115685, 90.90472698, 90.06366539, 88.35804987, 87.95780754, 88.53793812, 88.2097106, 89.5018487, 90.47539806, 90.60098362, 91.18615818, 91.73876047]
    right_ear_y = [93.29953539, 98.37297142, 99.54752475, 99.67029789, 102.150913, 22, 22, 22, 22, 112.4484177, 112.7293429, 117.7702496, 118.3558903, 117.3265737, 118.926703, 116.2423969, 119.2660415, 116.1094493, 116.5504321, 118.3850055, 120.3138404, 120.5849686, 120.8018575]

    print(DLC_position_preference(total_y_pixel=186, left_ear=left_ear_y,
                                  right_ear=right_ear_y, nose=nose_y
                                   ).position_preference())
