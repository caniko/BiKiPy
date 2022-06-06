===================
The Sequence Method
===================
Designed for experiments that can be segregated into sequential trial-sets.

Rules
=====
- Delimiting is with a dash, "-". Example: 1-Training. Reminder to delimit -> (delimit!)
- The tracking data is segregated into trial-sets. A trial-set consists of a sequence of trial tracking files. The trial tracking file has the sequence index stored in as a prefix in the file-stem as a number (delimit!). Optionally, for improved readability you can store a sequence label followed by the stage index. Example: 0-Habituation.h5, 1-Training.h5, 2-Test.h5.
- Meter pixel ratio must be defined
   - Define it yourself, and plug it into :code:`sequence_generate_configuration()`.
   -  If the experiment is in a confined box, or you know the length of a temporally fixed line in your video:

      #. Grab a video frame from one of the trial videos
      #. Define the line in MakeSense
      #. Make "Perimeter" directory in the base folder if it doesn't already exist.
      #. Export as csv and store in the "Perimeter" directory as "meter_pixel_ratio_{meter_length}.csv"; where meter_length is the length of the line in meters.

Structure
=========
- The dataset must be stored in the "dataset" directory
- Each trial-set is stored in a directory prefixed with the animal ID (delimit!).
- Optional, trial-set metadata; yaml format. Stored inside trial-set directory. Fields in metadata:
   - Optional, perimeter_set. Example: perimeter_set: A
- Project metadata; .xlsx or .odt, xlsx has best support (apologies to FOSS). The table must be in the sheet that is on index 0! The metadata file is stored on the root/base folder.
   - Animal ID column name must be "Animal"
   - Genetic state column must have the name "Gene"
   - Optional, "Cohort"
   - Optional, "Sex"
   - Optional, store the usage of a perimeter "Perimeter_{label_of_perimeter}". Row must be empty if there is no perimeter. Row must define the label to apply to the perimeter.
   - Make sure your dataset has no junk/invisible characters that might lead to problems with recognizing tags and performing comparisons.
- Only MakeSense perimeters are supported. These are stored in the "Perimeter" directory/folder.
   - The file-stem is the perimeter set ID (PID). If a perimeter set is stored in several files you must also include a unique identifier (UID) after the perimeter set ID (delimit!). Opinion: The unique identifier could be a sequence of numbers, letters, or random. Example: A-1.csv; where A is the perimeter set ID and 1 is the unique identifier.
   - Make sure that you don't use the same label for the different perimeters when they are defined in MakeSense. You can change the label in the file if you have to ensure this later.
   - Optional, for inspection, you can include an image with the perimeter set as the file-stem. Optionally, include the uid if it is specific to the subset (delimit!).
