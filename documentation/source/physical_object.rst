===============
Physical object
===============

Experiments with physical objects in them require definition. These definitions are used to analyse the overall interaction between the subject, and the individual object; additionally, analysis that include the interaction with several objects is also possible.


.. note::
    Experiments that support physical object analysis can be listed using the CLI, :code:`bkpy list physical-object`


Qualia definition
=================
The following ignores the hard problem of consciousness, and proceeds to the software modulation of annotation by human agents.

Method categories
-----------------
Qualia definitions from kinematics data requires axioms in two categories, ray-casting and proximity. Ray-casting axioms includes casting ray(s) from body region, and can help us determine when an object is within the field of view (FOV), or if it stimulates untraceable appendages like whiskers. The proximity axioms lets us decide a range of distance between a body region and the physical object; it will True if and only if the distance is within the defined range.

Combining these two axioms can define when an object is close enough (proximity), and simultaneously in the animals field of view (ray-cast). The two axioms by themselves would give an un-trivial amount of false positives.

Filtration method
-----------------
While every axiom is entirely spatial, they can also be modeled for temporal tuning. Read more about the tolerance model, and how it removes qualia-jitter and adds knobs for improved analysis.

In practice, the tolerance model modifies the data by adding a warmup phase and have a latency phase before triggering again.

Do we want one axiom's result to feed-forward to the next axiom, or do we want to consider each axiom individually before merging the results? These paths yield different results, and should be considered individually and comparatively.

In summary, there are two choices: Apply the axiom to the raw data individually, or filter the raw data sequentially through several axioms.
