######################################################################
This is our code for the bachelor project.

Libraries that are required in order to run the code are the following:
Numpy
Matplotlib.pyplot
CV2
Counter
struct
sys
sklearn

NOTE: It is important that the correct python3 interpreter is used.

######################################################################
Running the code:

1. Make sure the correct path is written when loading the MNIST dataset.
2. Open terminal and navigate to /code
3. Run this - Assuming pip is installed.

pip install -r requirements.txt
OR
python3 -m pip install -r requirements.txt

NOTE: This should install all necessary packages that are used in this code.
Make sure that it is installed to the correct interpreter of python.

4. In the terminal write python3 (assuming python3 is installed), else write python
followed by the path to the file you wish to run. An example is shown below:

python3 /PATH/TO/FILE/M4.py

Or simply

python3 M4.py

#################### IF YOU WANT TO RUN FEATURES ###################
1. Navigate to features in terminal
2. If all packages are installed correctly run

python3 filename.py

####################################################################

If running python3 M4.py results in errors saying that certain
packages were not found, then the packages might be installed to
the wrong python3, or the program is called using a different installation
of python3. If this is the case then run the following code in ther terminal

1. Find out what python3 is being used by running the command in the terminal:

which python3

2. Run the following command in the terminal

/opt/homebrew/bin/python3 -m pip install numpy matplotlib opencv-python scikit-learn seaborn

####################################################################

If the above fails, the solution is to create a virtual environment.

1. Navigate to the project root (/code)
2. Run in terminal: /opt/homebrew/bin/python3 -m venv venv
3. Activate venv: source venv/bin/activate
4. Run: pip install -r requirements.txt
5. Run a program by writing: python3 projectname.py
6. To deactivate venv write: deactivate
