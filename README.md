# Spectroscopy
Machine Learning Project where we compare models and construct ensembles using Spectroscopic datasets

## Abstract
[To be filled...]

If something does not work, read the section [below](https://github.com/techGIAN/Spectroscopy/edit/main/README.md/#it-does-not-work)

## Basic Requirements
Here are some basic requirements you need to have to run this:
1. Python (not sure if 2 will work but I recommend 3; I used 3.8)
2. Pandas (for handling data and dataframes)
3. Numpy (for mathematical and numerical operations)
4. Matplotlib (for plotting)
5. Scikitlearn (for machine learning models)
6. Keras (for neural networks)
7. Tensorflow (for computational speedup of NN models)
8. Most other packages are built-in with Python and simple imports are possible
9. ...

## Usage
First, navigate to the directory ```dataset```, where it contains a textfile describing where to download the dataset. And then navigate to the directory ```code```, and run (without 'user:~$'):
```console
user:~$ python3 preprocess.py --dataset [path_to_dataset] --seed [seed]
```
where ```path_to_dataset``` is the path to where you saved the downloaded dataset (preferably saved in the ```dataset``` directory), and ```seed``` is the seed value (```123``` by default). This creates a directory of datasets in ```datasets``` directory, which are the datasets we get from Step 1 after it has been partitioned to training, testing, etc. <br>

Next, run the models for each of the machine learning algorithms and each of the element. <i>(See more details below).</i> <br>

Now navigate to the directory ```code```, and run
```console
user:~$ python3 ensemble.py --ann [X] --cnn [X] --enets [X] --lasso [X] --linreg [X] --pls [X] --ridge [X] --svr [X] --xgb [X] --element [E]
```
where ```[E]``` is the element from one of ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O'], and ```[X]``` is the output ID result of the specified model. This produces an output in the ```results``` directory (you do not have to create this as this is done automatically when running the models) called ```ensemble_result.txt```, where it gives a summary table of the results -- a well as the ensemble model and its output.

### Running the Machine Learning Models (Non-Neural Networks)
This describes how to run the machine learning models, but this is for non-neural network models (see below for neural network models). These family of models are developed from ```scikitlearn``` libraries. <br>

Firstly, navigate to the ```code``` directory and then run:
```console
user:~$ python3 main.py --model [model] --params [path_to_parameters_file] --element [element] --seed [seed]
```
where ```element``` is the element from one of ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O'], ```seed``` is the seed value (```123``` by default), ```model``` is one of the following ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb'], and ```path_to_parameters_file``` is the path to the parameter file (found in the ```params``` directory). <br>

So for example, if you wish to run PLS on SiO2, then you run this command while inside the ```code``` directory:
```console
user:~$ python3 --model pls --params ../params/pls_params.txt --element SiO2 --seed 123
```
The output will be two files and they are both found in the directory ```results/M```, where ```M``` is a non-NN model found in ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb']:
1. ```outXXX_element=SiO2_model=pls.txt``` (the model results and results of the best-performing parameters for the model)
2. ```outXXX_element=SiO2_out_params=pls.txt``` (the parameters of the best-performing parameters for the model)
where ```XXX``` here is the three-digit ID of the output result (take note of the ```XXX``` here because this is the one you use for running the ```ensemble.py``` later on). <br>

You would have to do this for all the other non-NN models for SiO2, with the exception that the results output directory changes and filename changes accordingly. And if you do this for all other elements, then ```SiO2``` here will also change accordingly.<br>

### Running Neural Network Models
How to run is not that different. These models (for ANN and CNN) do not rely on ```scikitlearn``` because it is very limited with neural networks and not that efficient. We use ```keras``` here, but to make things faster, we employ ```tensorflow```. <br>

So again navigate to the ```code``` directory and then run the following:
```console
user:~$ python3 main-tf.py --model [model] --params [path_to_parameters_file] --element [element] --seed [seed] --model_id [m_id]
```
which is exactly the same as the one for running the non-neural network machine learning algorithms, except we use ```main-tf.py``` here and that we have an extra parameter. The ```m_id``` here is only relevant for CNNs. You can leave the parameter out for ANN. Two output files will be generated into the ```results/ann/```  directory when we run the ANN model. So it's not different from others. For CNN however, it's a little different. <br>

So, for CNN you need to specify the ```m_id```. Its value is from [1,2,3,4,5]. You need to run the CNN model five times on a specific element (unlike the other models that you only run once). For the first run ```m_id``` is 1, in the next run, it's 2, and so forth until you reach 5. Make sure that each run is successful (it will a display a success message on your terminal or console), before moving to the next run of the next ```m_id``` value. If for some reason, you made a mistake or there was an error when running say the 3rd run, then you need to go back to ```m_id``` having value 1 (i.e. the first run). So it has to be that CNN runs successfully five consecutive times each with their distinct model IDs from 1 to 5. To verify that this step is done correctly, make sure you have the following files in your ```results/cnn/``` directory generated:
1. ```outXXX_element=E_model=cnn_mod_id=I.txt```
2. ```outXXX_element=E_out_params=cnn_mod_id=I.txt```
where ```XXX``` is the result ID, ```E``` is the element, ```I``` is the ```m_id``` ([1,2,3,4,5]). And there will be 5 pairs of those in that directory (total of those 10 files) for that specific element ```E```. Take note of the result IDs of those 10 files. Take the minimum result ID of those 10, and take note of it and call it ```N``` (which we use when we compile). <br>

And then we compile this as follows (only do this for CNN). Still in the ```code``` directory, run
```console
user:~$ python3 compile.py --out [N] --element [E]
```
where ```E``` is the element and ```N``` is the ```N``` value obtained earlier. This consolidates the CNN results into a single file, whose output is also going to be generated in the ```results/cnn/```. Note that this compilation cannot be done unless all five CNN runs are consecutively successful.

## It Does Not Work...
If it does not work, here are some options:
1. Check the filenames, spelling-wise and sensitive-case-wise.
2. Check the directory you are working in.
3. Check if you have all the Python packages installed.
4. Contact me (last resort), with a detailed description of the error, and which part of the procedure above you are stuck in. I'll try my best to replicate the error and to help.

## Other Things To Do
- [x] Complete the restructured model.
- [ ] Test on SiO2 on another computer (note it has only been tested on SiO2 on my computer)
- [ ] Test on all other 7 elements on both computers.
- [ ] Plot ternary plots
- [ ] Plot actual-vs-observed
- [ ] Runtime analysis
- [ ] ...
