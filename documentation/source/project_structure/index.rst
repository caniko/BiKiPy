==================
Project Structures
==================
There are many kinds of experiment designs. The data from these experiments requires a standard for storage; otherwise, each data structure would require its own reading strategy to be used with BiKiPy. Each structure is named in a way that makes sense to the user.


.. toctree::
   :titlesonly:
   :maxdepth: 0

   sequence


Global Rules
============
- Delimiting in file or directory names is always with a dash, "-", 1-Training.h5.
- The experimental dataset must be stored in the "dataset" directory

Settings
--------
Each project defines a settings.yaml file that exposes parameters for data ingress, analysis, and export. Some of them are required while most are optional.

The following sub-section define setting parameters that are universal across project structures, and how to define them.

Meter pixel ratio
~~~~~~~~~~~~~~~~~
The meter pixel ratio must be defined to convert pixel distances to metric units after project initialization. Supported methods:

- Can be defined manually in the settings.yaml file.
- Line segment from video:
   #. Grab a video frame from one of the trial videos
   #. Define the line in MakeSense
   #. Make "Perimeter" directory in the base folder if it doesn't already exist.
   #. Export as csv and store in the "Perimeter" directory as "meters_per_pixel_{meter_length}.csv"; where meter_length is the length of the line in meters.
- TODO: The metric units of the video resolution can be approximated

Perimeter
---------
- Only MakeSense perimeters are supported. These are stored in the "Perimeter" directory/folder.
- The file-stem is the perimeter set ID (PID). If a perimeter set is stored in several files you must also include a unique identifier (UID) after the perimeter set ID (delimit!). Opinion: The unique identifier could be a sequence of numbers, letters, or random. Example: A-1.csv; where A is the perimeter set ID and 1 is the unique identifier.
- Make sure that you don't use the same label for the different perimeters when they are defined in MakeSense. You can change the label in the file if you have to ensure this later.
- Optional, for inspection, you can include an image with the perimeter set as the file-stem. Optionally, include the uid if it is specific to the subset (delimit!).

Animal Metadata
---------------
An excel sheet; either metadata.xlsx or metadata.odt. The table must be in the sheet that is on index 0! The metadata file is stored on the root/base folder. We use this file to store data about each animal. The data is defined column-wise

Stageful multi indexing
~~~~~~~~~~~~~~~~~~~~~~~
You may multi-index the columns to include stages. This is useful if some features vary across experiment stages. The second header of the column must be the stage index. The 2nd header should be labeled with "All" if it is a constant across all stages.

.. table::
   :align: center

   ===========  ===========  ===========
       Sex              Perimeter
   -----------  ------------------------
       All           0            1
   ===========  ===========  ===========
        M       Perimeter A  Perimeter D
        F       Perimeter B  Perimeter A
        M       Perimeter A  Perimeter A
        M       Perimeter C  Perimeter B
   ===========  ===========  ===========

Supported Columns
~~~~~~~~~~~~~~~~~
- Animal ID column name must be "Animal"
- Optional, label of perimeter where "perimeter-{label_of_perimeter}". Row must be empty if there is no perimeter.
- Any generic feature, such as the genotype, can be included in its own column. These features will be added to the result of the respective animal ID. Column names such as: "Gene", "Cohort", "Sex".
- Make sure the dataset has no junk/invisible characters that might lead to problems with the recognition of the tags.
