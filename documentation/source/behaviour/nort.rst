=============================
Novel object recognition test
=============================
:code:`bikipy.behaviour.nort` is the submodule that stores the analysis pipeline devised for novel object recognition test.

NORT objects
============
The NORT objects need to be annotated, and the annotation data has to be stored as :code:`bikipy.behaviour.nort.trial.NortField` object. The NortField 

Observation
===========
Each object must have their number of observation instances across frames determined in order to determine cases of recognition. :code:`nort_observation` was designed specifically for this problem. It takes the following arguments:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Variable name
     - Description
   * - nort_object
     - The object class must inherit from :code:`bikipy.perimeter.base.PolygonalPerimeter`, more information can be found here.
   * - nose
     - Sequence with the coordinates pointing to the location of the **nose** on each video frame during the trial
   * - eye_center
     - Sequence with the coordinates pointing to the location of the **eye center** on each video frame during the trial
   * - torso
     - Sequence with the coordinates pointing to the location of the **torso** on each video frame during the trial
   * - fps
     - Frames per second (fps) of the trial video recording
   * - perimeter_border_normal_pixel_magnitude
     - The normal pixel distance between the border and the respective object
   * - maximum_radians_inter_gaze_perimeter
     - Maximum radians between the gaze vector (eye_centre to nose) and object tangent
   * - inspect (default :code:`False`)
     - If :code:`True`, will generate and show and inspection figure for the inspection of each filter


