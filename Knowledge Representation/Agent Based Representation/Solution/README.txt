-------------------------------------------------------------------------------
Knowledge Representation
Assignment 1 - Agent-based representation of knowledge
Michael McAleer (R00143621)
-------------------------------------------------------------------------------
Requirements:
-Python 3.7
-iPython Blocks (can be ignored if not using iPython Notebook file)
-iPython (can be ignored if not using iPython Notebook file)

Notes:
-The three parts of the assignment can be run directly from the command line
using $ python <file_name>.py  no other input is required

-All files are created to be run from within their source directory, there is
no need to copy them into AIMA repo first, the required classes are imported
from aima_src.py

-There have been multiple sub-classes implemented within each part, but on
occassion bugs were fixed within the source classes, an example of this is
random location selection, the calculation of outer bounds was incorrect and
required an addition of -1 to the code to work correctly.

-There is a preference over well commented and structured code instead of
Jupyter notebooks or iPython. The code reads very easily itself and at all
stages the functionality is accurately and concisely annotated. The iPython
notebook file is only present to cover the visual representation element of the
assignment.

-For part-3 of the assignment, there is a parameter which can be changed at the
bottom of the file to determine if breadth-first or depth-first should be used.
To change between either search algorithm, change the search_type to 'bfs' or
'dfs' as required.  For details on what default_moves does to the run, please
see the report. Setting this to True will maintain traditional search algorithm
behaviour.