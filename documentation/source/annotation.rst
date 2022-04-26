==========
Annotation
==========
Before we start, we need to get the frame(s) from our video(s). We use VLC_; it is very easy to get `snapshots with VLC`_. Try to take a snapshot where the object(s) to be annotated is clearly visible.

`MakeSense`_ annotation is natively supported by BiKiPy:

#. On the homepage of MakeSense, select "Get Started"
#. Select "Object Detection"
#. Label names should be the correct values with respect to the experiment. In a NORT novel session it would be:
    * Training: "Variable" (will be replaced with novel NORT session), "Constant"
    * Novel: "Novel" (switched with variable), "Constant" (optional)
#. Select "Polygon", and start annotating. Every time you are done annotating an object, you must remember to assign the label to your object.
#. Export labels as COCO dataset, and save at the project directory.

Label names can be found in the respective experiment section in the documentation.
.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/7Wrw36hscrY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



.. _VLC: https://www.videolan.org/vlc/
.. _snapshots with VLC: https://wiki.videolan.org/Documentation:Snapshots/
.. _MakeSense: https://www.makesense.ai/
