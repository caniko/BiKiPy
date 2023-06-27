.. _qualia:
============================================
Physical object attentive observation qualia
============================================
The following ignores the hard problem of consciousness, and utilizes heuristics that are inspired from expert annotation agents exclusively.

Qualia heuristics help us automatically define instances of attentive physical object observation. We employ axioms from two categories, in line-of-sight (iLOS) and proximity, to define qualia heuristics. iLOS axioms includes casting rays from a body region and detecting if the rays collide with the physical object, ray to perimeter collisions mean the object is in the purview of the body region. Proximity axioms lets us threshold distances from physical object to the animal, allowing us to filter frames where the animal is too far away from the object for the respective heuristic.

At the discretion of the end-user, heuristics can also be combined using AND or OR logic; the resulting boolean index is treated as an additional result.

.. note::
   Results from qualia heuristics are boolean indices, and can be tolerance modeled.

Heuristics
==========
Qualia heuristics are pre-defined, and must be assigned to a project through the BiKiPy UI.

Proximal field of view profile (pFOV)
-------------------------------------
The animal ears are tracked separately, and we apply two axioms to each:

- The ear has to have a certain proximity to the object
- Rays cast within certain radial range must collide with the physical object

After applying this axioms, and merging them into one sequence with logical AND, we reduce the resulting sequence with logical OR to get our final observation boolean index, pFOV.

Olfaction profile
-----------------
The olfaction heuristic considers the proximity of the nose region to the nose in addition to a iLOS ray-cast from the nose.

.. note::
   Consider merging the pFOV and olfaction profiles for a holistic rodent sensation profile

Addendum heuristics
-------------------
Some heuristics don't work by themselves, and perform an assistive role to complete heuristics.

Objects defined by penetrable perimeters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Should the object perimeter be accessible, i.e. the animal can be confined on top of the object, it is assumed that the animal isn't attentive to the object—rather uses it as a platform for other actions. These instances are defined by tracking the body-mass representative region like the torso as confined inside the object perimeter, and set these instances to FALSE in the boolean index.

Physical object observation metrics
===================================
Consider that we are using heuristics to achieve the observation boolean indices for each object.

Seconds observing object
------------------------
The sum of the object observation boolean index divided by the frame per second.

Total seconds observing
-----------------------
We reduce the boolean indices of every object observation with logical OR, and perform the sum the boolean index and divide by the frames per second. Similar to seconds observing object.
