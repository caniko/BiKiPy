================
Circle perimeter
================
Confinement in circle perimeters are computed by (1) computing the distance from the circle center to the provided coordinates:

.. math::
    d_t = | s_t - c |

Where :math:`s_t` is the coordinate of the region of interest in a given time; :math:`c` is the coordinate of the circle center; :math:`d_t` is the distance from the circle center.

Finally, (2) we threshold the distance:

.. math::
    inCircle_t = d_t < r

Where :math:`r` is the radius of the circle.

We do this for all the coordinates in the given trial giving us a sequence of booleans stating the binary confinement of the animal.
