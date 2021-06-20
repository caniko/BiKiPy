Features is a subpackage that stores functions for the computation of behavioural features. These functions can be integrated into :code:`trial`.

Angle
-----
The angle between key parts of the animal body can be used to track functions such as balance and states of focus. Moreover, there are two methods to go about computing angle.

:code:`feature.angle.inner_angle` computes the inner angle between two vectors that intersect. The solution is based on the definition of the dot product:

.. math::
    \\theta = cos^{-1}(\\mathbf{a} \\cdot \\mathbf{a})
    Where \\theta

:code:`feature.angle.clockwise_angel_2d` computes the angle in the clockwise direction no matter what.

Midpoint
--------

