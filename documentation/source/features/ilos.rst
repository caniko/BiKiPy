================
In line of sight
================

The in line-of-sight (iLOS) problem allows us to define when objects are in the animals purview.

Polygon
=======
Solving the iLOS problem for polygons with more than three sides have no general trivial solution. The existing state of the art method is to solve computationally using by casting rays from the origin of interest, and check for collisions with regions of interest.

We need two points of reference to define the direction of the rays. One of the points is always the origin of interest; however, the second point must be defined at the discretion of the designer and is arbitrary.

.. note::
   Considering top-down recordings, the second point for a rodents nose could be the center of the ears; creating a ray that goes through the snout and exiting through the nose.
