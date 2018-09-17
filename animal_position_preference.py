#from Vector2D import Vector2D


class AnimalPositionPreference:

    def __init__(self, max_y_pixel, left_ear, right_ear, nose):
        self.max_y_pixel = max_y_pixel    # example: for 388x188, max_y_pixel = 186
        self.left_ear = left_ear  # list of y coordinates for body part nr. 1
        self.right_ear = right_ear  # list of y coordinates for body part nr. 2
        self.nose = nose  # list of y coordinates for body part nr. 3l
    
    @property
    def two_cm_upper_limit(self):
        """ Finds horizontal line 2 cm from the bottom line and 2 cm from the upper line. """
        return self.max_y_pixel / 4.65

    @property
    def two_cm_lower_limit(self):
        """ Finds horizontal line 2 cm from the bottom line and 2 cm from the upper line. """
        return self.max_y_pixel / 1.18

    def position_preference(self):
        lower_environment = 0
        upper_environment = 0
        total_nr_of_frames = 0
        """
        for y1, y2, y3 in self.left_ear, self.right_ear, self.nose:
            if y1 and y3 or y2 > self.two_cm_lower_limit:
                lower_environment += 1
            elif y2 and y3 or y1 > self.two_cm_lower_limit:
                lower_environment += 1
            elif y1 and y3 or y2 < self.two_cm_upper_limit:
                upper_environment += 1
            elif y2 and y3 or y1 < self.two_cm_upper_limit:
                upper_environment += 1
            total_nr_of_frames += 1"""

        for i in range(len(self.nose)):
            test_ear_lower_env = self.left_ear[i] > self.two_cm_lower_limit\
                                 + self.right_ear[i] > self.two_cm_lower_limit
            test_ear_upper_env = self.left_ear[i] < self.two_cm_upper_limit\
                                 + self.right_ear[i] < self.two_cm_upper_limit

            print(self.two_cm_upper_limit)

            if test_ear_lower_env and self.nose[i] > self.two_cm_lower_limit:
                lower_environment += 1
            elif test_ear_upper_env and self.nose[i] < self.two_cm_upper_limit:
                upper_environment += 1
            total_nr_of_frames += 1

        percent_upper = (upper_environment / total_nr_of_frames) * 100
        percent_lower = (lower_environment / total_nr_of_frames) * 100
        return ("Lower environment: {}% \nUpper environment: {}%".format(percent_lower, percent_upper))


nose_y = [158, 74.8017916, 72.76470613, 71.83685446, 72.06032467, 77.660725, 80.501369, 84.9468677, 86.4281106, 84.80722725, 88.91118813, 95.83561897, 98.05349469, 97.35266972, 99.20710927, 98.69586754, 99.39605695, 96.12210178, 98.44720972, 98.35878849, 104.328619, 104.5202456, 104.4529443]
left_ear_y = [158, 85.095456, 80.73817682, 78.80043721, 78.64884067, 81.50397062, 86.85195661, 87.50960016, 87.39562154, 87.63862348, 86.60903049, 89.96115685, 90.90472698, 90.06366539, 88.35804987, 87.95780754, 88.53793812, 88.2097106, 89.5018487, 90.47539806, 90.60098362, 91.18615818, 91.73876047]
right_ear_y = [93.29953539, 98.37297142, 99.54752475, 99.67029789, 102.150913, 106.2793674, 110.2543111, 111.2428942, 111.8114631, 112.4484177, 112.7293429, 117.7702496, 118.3558903, 117.3265737, 118.926703, 116.2423969, 119.2660415, 116.1094493, 116.5504321, 118.3850055, 120.3138404, 120.5849686, 120.8018575]

print(AnimalPositionPreference(max_y_pixel=186, left_ear=left_ear_y, right_ear=right_ear_y, nose=nose_y
                               ).position_preference())