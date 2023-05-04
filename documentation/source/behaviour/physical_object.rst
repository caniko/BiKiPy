===============
Physical object
===============

The following ignores the hard problem of consciousness, and proceeds to the software modulation of annotation by human agents.

Qualia definitions from kinematics data requires axioms in two categories, ray-casting and proximity. Ray-casting axioms includes casting ray(s) from body region, and can help us determine when an object is within the field of view (FOV), or if it stimulates untraceable appendages like whiskers. The proximity axioms lets us decide a range of distance between a body region and the physical object; it will True if and only if the distance is within the defined range.

We combine axioms to define the occurrences of observation qualia at a given time, these combination are referred to as qualia profiles.

.. note::
   Qualia profiles are pre-defined, and must be assigned to a project through the BiKiPy UI.

.. note::
   Qualia profiles can be tolerance modeled!

At the discretion of the end-user, profiles can also be combined using AND or OR logic.

Proximal field of view profile
==============================
The animal eyes are tracked separately, and we apply two axioms to each:

- The eye has to have a certain proximity to the object
- Rays cast within certain radial range must collide with the physical object

After applying this axioms, and merging them into one sequence with logical AND, we reduce the resulting sequence with logical OR to get our final qualia profile.

Additional
----------
Should the object perimeter be penetrable, i.e. the animal can be confined on top of the object. We may check for instances of a certain body-mass representative region like the torso as confined outside the object, and set these instances to FALSE in the qualia profile as the animal is not directly observing the object.
