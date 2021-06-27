==========
Annotation
==========
Before we start, we need to get the frame(s) from our video(s). We use VLC_; it is very easy to get `snapshots with VLC`_. Try to take a snapshot where the object(s) to be annotated is clearly visible.

`Make Sense`_ annotation is natively supported by BiKiPy:

#. At the website, select "Get Started"
#. Select "Object Detection"
#. Label names should be:
    * Training: "Variable" (will be replaced with novel NORT session), "Constant"
    * Novel: "Novel" (switched with variable), "Constant" (optional)
#. Select "Polygon", and start annotating. Every time you are done annotating an object, you must remember to assign the label to your object.
#. Export labels as COCO dataset, and save at the project directory.


.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/7Wrw36hscrY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



.. _VLC: https://www.videolan.org/vlc/
.. _snapshots with VLC: https://wiki.videolan.org/Documentation:Snapshots/
.. _Make Sense: https://www.makesense.ai/