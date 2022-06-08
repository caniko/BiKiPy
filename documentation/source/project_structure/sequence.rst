===================
The Sequence Method
===================
Designed for experiments that can be segregated into sequential trial-sets. The Sequence structure should be used when each animal goes through a sequenced set of trials: 0-Habituation => 1-Training => 2-Test.

Trial dataset
-------------
Each trial-dataset is stored in a directory prefixed with the animal ID (delimit!). These datasets consist of tracking files that are sequentially prefixed (delimit!). Optionally, you can store a sequence label followed by the stage index, makes it easier for outsiders to understand. Example: 0-Habituation.h5, 1-Training.h5, 2-Test.h5.

Meter pixel ratio
-----------------
The meter pixel ratio must be defined to convert pixel distances to metric units.

Definition
~~~~~~~~~~
- Manually, and plug it into :code:`sequence_generate_configuration()`.
- Line segment in the video:
   #. Grab a video frame from one of the trial videos
   #. Define the line in MakeSense
   #. Make "Perimeter" directory in the base folder if it doesn't already exist.
   #. Export as csv and store in the "Perimeter" directory as "meter_pixel_ratio_{meter_length}.csv"; where meter_length is the length of the line in meters.
- TODO: The metric units of the video resolution can be approximated

Perimeter
---------
Only MakeSense perimeters are supported. These are stored in the "Perimeter" directory/folder.

- Optional, for inspection, you can include an image with the perimeter set as the file-stem. Optionally, include the uid if it is specific to the subset (delimit!).

Metadata
========
Project metadata; .xlsx or .odt, xlsx has best support. The table must be in the sheet that is on index 0! The metadata file is stored on the root/base folder.

Columns
-------
- Animal ID column name must be "Animal"
- Optional, label of perimeter where "Perimeter_{label_of_perimeter}". Rules:
   - Row must be empty if there is no perimeter.
   - Sub-columns or multi-indexed columns can be used to define perimeters across experiment stages. The header of the column must be the stage index.

.. table::
   :align: center

   ===========  ===========
           Perimeter
   ------------------------
        0            1
   ===========  ===========
   Perimeter A  Perimeter D
   Perimeter B  Perimeter A
   Perimeter A  Perimeter A
   Perimeter C  Perimeter B
   ===========  ===========

- Any generic feature, such as the genotype, can be included in its own column. These features will be added to the result of the respective animal ID. Column names such as: "Gene", "Cohort", "Sex".
- Make sure the dataset has no junk/invisible characters that might lead to problems with the recognition of the tags.
