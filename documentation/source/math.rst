====
Math
====
The math package stores auxiliary functions that are mathematical. Functions mentioned here are functions that might be useful for developers who are looking for functions that may be directly useful in their own bikipy pipeline.


Point in parallelogram
======================
Function path :code:`bikipy.math.point_in_parallelogram.point_in_parallelogram`.

Let :math:`ABCD` be our parallelogram, :math:`A` the origin of our coordinate system. Then, the vectors :math:`\overrightarrow{AB} = \mathbf{a} = (a_x, a_y)` and :math:`\overrightarrow{AD} = \mathbf{b} = (b_x, b_y)` define the parallelogram. Then, the vectors :math:`\mathbf{a'} = (-a_y, a_x)` and :math:`\mathbf{b'}=(-b_y,b_x)` are orthogonal to :math:`\mathbf{a}` and :math:`\mathbf{b}`. Moreover, we need to make sure that they are in the right direction. We can do that with the dot product:

.. math::
    \mathbf{a''} = \operatorname{sign}(\mathbf{a'} \cdot \mathbf{b}) \mathbf{a'}, \quad
    \mathbf{b''} = \operatorname{sign}(\mathbf{a} \cdot \mathbf{b'}) \mathbf{b'}

All :math:`p` satisfying

.. math::
    0 \le \mathbf{a''} \cdot p \le \mathbf{a''} \cdot \mathbf{b}

lie in the area between the parallel line segments :math:`AB` and :math:`CD`, and the :math:`p` satisfying

.. math::
    0 \le \mathbf{b''} \cdot p \le \mathbf{b''} \cdot \mathbf{a}

lie in the area between the line segments :math:`AD` and :math:`BC`.

The two equalities put together are the criterion for :math:`p` being in the area spanned by the parallelogram.

Rectangle
---------
There is also a special case in which the parallelogram is a rectangle as there is a faster way for computing the solution in this special case:

.. math::
    \mathbf{\hat{a}''} = \mathbf{\hat{b}} \\
    \mathbf{\hat{b}''} = \mathbf{\hat{a}}
