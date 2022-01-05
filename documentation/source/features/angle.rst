=====
Angle
=====
The angle between key parts of the animal body can be used to track functions such as balance and states of focus. Moreover, there are two methods for computing the angle.

Inner angle
-----------
:code:`feature.angle.inner_angle` computes the inner angle between two vectors that intersect. The solution is based on the definition of the dot product:

.. math::
    \theta = \cos^{-1} \bigg( \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| |\mathbf{b}|} \bigg) \quad \theta \in [0, \pi]

Clockwise angle
---------------
:code:`feature.angle.clockwise_angel_2d` computes the angle in the counterclockwise direction, using the definition of the determinant and the dot product along with the atan2_ function:

.. math::
    \theta = \pi + \operatorname{atan2} (\det(\mathbf{\hat{b}}, \mathbf{\hat{a}}), \mathbf{\hat{b}} \cdot \mathbf{\hat{a}})

Where :math:`\theta \in [0, 2\pi]`; :math:`\mathbf{\hat{a}}` is in the starting direction; :math:`\mathbf{\hat{b}}` is in the ending direction.


.. _atan2: https://en.wikipedia.org/wiki/Atan2
