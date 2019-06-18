Structure of the ZIP File:

- PW3-SEL-1819-Team2/ folder: Python project containing the dataset, the documentation and the python code (Pycharm: Open Project)

	- Documentation/ folder:
		- SEL_Final_Project.pdf: Pdf containing the report of the practical work.

	- Data/ folder: Contains the 3 datasets for the Case Base Recommender System
		- 'all_songs.csv'	--> File containing all available songs and their features
		- 'songs_test.csv'	--> File containing all cases allocated to testing
		- 'songs_train.csv'	--> File containing all cases allocated to training
		
	- Source/ folder: Contains the python code of the practical work
		- main.py: Main file to run the program. The user can modify the parameters of the program at the beginning of this file.
		- case_base.py: It contains the implementation of the Case-Based Recommender System with its four stages: retrieve, reuse [configuration, planning], revise and retain
		- spotify_api.py: It contains the values that describe the histograms for the acousticness, loudness, tempo, ... as well as functions to normalize those values.
		- utils/ Folder:
			- preprocessing.py: It contains the necessary methods to preprocess the dataset.
			- aux_functions.py: It contains auxiliary functions such as the menu, euclidean distance, etc.
			- node.py: It contains the Node class for building the Case Discrimant Tree.