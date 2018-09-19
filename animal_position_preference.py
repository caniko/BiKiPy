from Vector2D import Vector2D


class AnimalPositionPreference:

    def __init__(self, y_pixels, y_left_ear, y_right_ear, y_nose):
        self.y_pixels = y_pixels    # example: for 388x188, y_pixels = 188
        self.y_left_ear = y_left_ear  # list of y coordinates for body part nr. 1
        self.y_right_ear = y_right_ear  # list of y coordinates for body part nr. 2
        self.y_nose = y_nose  # list of y coordinates for body part nr. 3

    @property
    def midline(self):
        """ Finds horizontal line at the middle of the environment.
            This is assuming that the whole video is only the
            environment where the animal can move. """
        return self.y_pixels / 2.0

    def position_preference(self):
        lower_environment = 0
        upper_environment = 0
        for y1, y2, y3 in self.left_ear, self.right_ear, self.nose:
            if y1 and y3 or y2 > self.midline:
                lower_environment += 1
            elif y2 and y3 or y1< self.midline:
                upper_environment += 1
        percent_upper = (upper_environment / (lower_environment + upper_environment)) * 100
        percent_lower = (lower_environment / (lower_environment + upper_environment)) * 100
        return ("Lower environment: {} \n Upper environment: {}".format(percent_lower, percent_upper))

