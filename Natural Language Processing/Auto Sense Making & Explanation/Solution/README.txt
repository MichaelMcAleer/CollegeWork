# -----------------------------------------------------
# Natural Language Processing
# Assignment 2 - Automatic Sense Making and Explanation
# Michael McAleer R00143621
# -----------------------------------------------------

The notebooks included in this archive assume that they are being run in the
same directory as the 'data' directory containing all training and test data
if running locally on Windows, or, that the data lives on Google Drive and that
path is mounted in Colab.

To adjust these settings edit the following values:

    if os.name == 'nt':
        ROOT_DIR = os.getcwd()
    # Else running on CoLab, set ROOT_DIR to match environment path
    else:
        from google.colab import drive

        drive.mount('/content/drive')
        ROOT_DIR = '/content/drive/My Drive/Colab Notebooks'

The above block of code is consistent throughout all files, change ROOT_DIR
to match the path to the directory containing the training and test datasets
in a parent folder called data.

Data Directory Layout

Data Dir
|--- Train Dir
|       |-----Train TaskA - Data & Answers
|       |-----Train TaskB - Data & Answers
|       |-----Train TaskC - Data & References
|--- Test Dir
        |-----Test TaskA - Data & Answers
        |-----Test TaskB - Data & Answers
        |-----Test TaskC - Data & References