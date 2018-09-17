import numpy as np


class Vector2D:
    """Class for representing two-dimensional vectors."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({:g}, {:g}".format(self.x, self.y)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__,
                                   self.x, self.y)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            x = self.x + other.x
            y = self.y + other.y
            return self.__class__(x, y)

        else:
            raise TypeError("cannot add vector and {}".format(type(other)))

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self + (-other)
        else:
            raise TypeError("cannot subtract vector and {}".format(type(other)))

    def dot(self, other):
        return self. x * other.x + self. y * other.y

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.dot(other)
        elif isinstance(other, (int, float)):
            return self.__class__(self.x * other, self.y * other)
        else:
            raise TypeError("cannot multiply vector and {}".format(type(other)))

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """Interpret u@v as angle between two vectors"""
        return np.arccos(
            (self * other)/self.length()*other.length()
        )

    def perpendicular(self, other):
        return np.isclose(self*other, 0)

    def two_points_to_vector(self, vector_origin_point, end_point):
        return self.__class__(vector_origin_point[0] - end_point[0],
                              vector_origin_point[1] - end_point[1])

    @property
    def length(self):
        return np.sqrt(self * self)

    @length.setter
    def length(self, new_length):
        scale = new_length/self.length
        self.x *= scale
        self.y *= scale

    def unit(self):
        """Return a unit vector with the same orientation."""
        if self.length == 0:
            raise RuntimeError("Vector of zero length has no unit vector.")

        new_vector = self.__class__(self.x, self.y)
        new_vector.length = 1
        return new_vector

    def __eq__(self, other):
        same_x = np.isclose(self.x, other.x)
        same_y = np.isclose(self.y, other.y)
        return same_x and same_y
