.. _physical-object:
===============
Physical object
===============

The following ignores the hard problem of consciousness, and proceeds to the software modulation of annotation by human agents using *qualia heuristics*.

Qualia heuristics help us automatically define instances of attentive physical object observation. We employ axioms from two categories, ray-casting and proximity, to define qualia heuristics. Ray-casting axioms includes casting rays from a body region and detecting if the rays collide with the physical object. Proximity axioms lets us threshold distances from physical object to the animal, allowing us to filter frames where the animal is too far away from the object for the respective heuristic.

At the discretion of the end-user, heuristics can also be combined using AND or OR logic; the resulting boolean index is treated as a heuristic.

.. note::
   Qualia heuristics are pre-defined, and must be assigned to a project through the BiKiPy UI.

.. note::
   Results from qualia heuristics are boolean indices, and can be tolerance modeled.

Heuristics
==========

Proximal field of view profile (pFOV)
-------------------------------------
The animal eyes are tracked separately, and we apply two axioms to each:

- The eye has to have a certain proximity to the object
- Rays cast within certain radial range must collide with the physical object

After applying this axioms, and merging them into one sequence with logical AND, we reduce the resulting sequence with logical OR to get our final qualia heuristic.

Object that are penetrable
^^^^^^^^^^^^^^^^^^^^^^^^^^
Should the object perimeter be penetrable, i.e. the animal can be confined on top of the object. We may check for instances of a certain body-mass representative region like the torso as confined outside the object, and set these instances to FALSE in the qualia heuristic as the animal is not directly observing the object.

Olfaction profile
-----------------
The olfaction heuristic considers the proximity of the nose region only, i.e. only a nose proximity axiom is used.

.. note::
   Consider merging the pFOV and olfaction profiles for a holistic rodent sensation profile
