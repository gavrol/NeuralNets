In this directory
1) main_basic.py:
	a main script that can be run on various data files in data-dir;
	this is a basic script; the embellished version of this main is 	 main_using_neuralnet.py;
	
2) main_using_neuralnet.py: 
	is the main that could be run on a data set of interest;
	it uses neuralnet.py and funtions.py

3) create_data_files.py: 
	script used to create different types of data which then could be modelled with neural networks

4) main_NN_4BooleanResponse.py:
	work in progress... using NN for predicting a categorical response variable;
	
Function scripts (are not run by themselves, but are called by the above mains):
a) neuralnet.py:
	it's the module that actually implements the "mechanics" of neural networks
b) functions.py:
	a set of functions callable by main_using_neuralnet.py