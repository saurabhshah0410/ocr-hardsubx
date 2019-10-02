# Improve the OCR Subsystem | CCExtractor
The aim of the project was to extract hardcoded subtitles from input videos (real-time) and generate SRT file without asking the user to input the color, duration, ocr mode, italicized subs etc. Upon completion of the project, the following objectives have been achieved:

* Feature to extract mutli-colored subtitles has been added, enhancing usage of CCExtractor to videos with multi-color subs
* Existing captioning module is now independent of arbitrary input parameters like color, confidence, luminance making it user-friendly
* Accuracy of the subs has been increased and noise has been reduced

## Compilation:
CCExtractor can be compiled with HardsubX support as follows: `make ENABLE_HARDSUBX=yes`
This needs to be run from the `ccextractor/linux` directory.

## Usage:
The `-hardsubx` flag needs to be specified to the ccextractor executable in order to enable burned-in subtitle extraction.
Other options such as `ocr_mode`, `subcolor`, `whiteness_thresh`` etc have been made optional as they'll now be self detected.

A composite example command is as follows:-
`ccextractor video.mp4 -hardsubx -subcolor white -detect_italics -whiteness_thresh 90 -conf_thresh 60`

## Week 1:
* Collecting different types of videos from the sample platform and other web sources like youtube etc to make a dataset of videos on which the code will be tested.
* Transcripts had to be generated for those videos which were already hardsubbed.
* The closed captions for the videos from youtube were downloaded using youtube-dl and burned into the corresponding videos using handbrake and ffmpeg, which included a lot of variety like multi colored subtitles, different font styles etc. The curated videos can be found [here](https://drive.google.com/open?id=1yHZN4jQw24MIgFGrByUbUZahkUelQHD8).
* Going through my proposal and the opencv code to get a clearer idea of what I would be doing in Week2.

## Week 2:
* Identifying the functions for text detection from the opencv implementation and filtering out those parts from that code in the `opencv` repository which are not of interest to my project.
* I've started to implement the functions from opencv by converting them from C++ to C(not completely syntactically correct though) so that they could be integrated into CCExtractor's codebase.
So far, I've identified the following structs and functions which I've implemented in C:
* struct `vector`: similar to std::vector in c++.
* struct `Mat` :  the basic struct which represents an image.
* Functions: `create`, `createVector`, `createusingrect`, `createusingscalar`, `createusingrange`: Initializers for `Mat`
* `computeNMChannels` : Extract the intensity, hue, saturation and intensity gradient channels from the frame.
* `copyTo`, `convertTo`, `cvtColor`, `get_gradient_magnitude`, `split` these functions serve as the helper functions to `computeNMChannels`.

 ## Week 3: 
* struct `ERStat`: represents the class-speicifc Extremal Regions.
* `loadclassifierNM`: allow to implicitly load the default classifier when creating an ERFilter object.
* `createERFilterNM1`: create an Extremal Region Filter for the 1st stage classifier.
* `createERFilterNM2`: create an Extremal Region Filter for the 2nd stage classifier.
* `evalNM1`, `evalNM2`: returns the probability measure for the regions as described in the algo.

## Week 4 & 5:
These 2 weeks are used for the implementation of the key function `run` and it's helper functions listed below. I'll also start the implementation of the `erGrouping()` function in this interval of 2 weeks.
* `run`: extracts the component tree and filter the ERs using a given classifier.
* `er_tree_extract`: extract the component tree and store all the ER regions.
* `er_add_pixel`: accumulate a pixel into an ER.
* `er_merge`: merge an ER with it's nested parent.
* `er_save`: copy extracted regions into the output vector.
* `er_tree_filter`: recursively walk the tree and filter(remove) regions using the callback classifier.
* `er_tree_nonmax_suppression`: recursively walk the tree selecting only regions with local maxima probability
*  `deleteERStatTree`: deletes a tree of ERStat regions starting at root.
* `erGrouping()`: find groups of ERs that are organized as text blocks.

## Week 6 & 7:
This period was used to complete all the previous pending work and many minor structs and helper functions for the remaining code. Now, a few major functions are left and I'll be able to complete the text detection part soon.
* `floodFill`, `floodFill_CnIR1`, `floodFill_CnIR3`, `floodFillGrad_CnIR1`, `floodFillGrad_CnIR3`: Complete implementation of floodfill algorithm.
* `findContours` and it's helper functions which will be used in the complete implementation of `erGrouping()`.
* Many structs and functions related to memory management implemented.
* Cleaning up the code, identifying and rectifying the bugs in the code written so far.

## Week 8, 9 & 10:
All the machine learning related functionality was implemented in this period.
* `load_ml`, `read_ml`, `predict_ml`: to load and obtain the results from the trained models `trained_classifierNM1.xml` and `trained_classifierNM2.xml`
* The text detector is complete by now and in the following weeks, I plan to implement the text recognition part.

## Week 11 & 12:
This period was used to implement the functions required for recognizing text from the text containing regions using `tesseract-ocr`.
* `run_ocr`: takes as input the binarized image and returns the text recognized from this image.
*  some minor structs and other helper functions were implemented required for the completion of the text recognition part.

## Week 13:
The final week was used to make the required changes in the codebase of CCExtractor and integrate my code into the hard subtitle extraction module.
