# Spectroscopy
Machine Learning Project where we compare machine learning models' performances evaluated on spectrum datasets and then constructing ensembles

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
1. Clone the repository. Run the following command (without 'user:~$'):
```console
user:~$ git clone https://github.com/techGIAN/Spectroscopy.git
```
2. Download the dataset: https://drive.google.com/drive/folders/1YfvpQyREb9LdMQ_pR9YLNn0c0YoaMq3z?usp=sharing. And then unzip it and place the unzipped file in the directory called ```dataset/```. <br>

3. Navigate to the ```code/``` directory using the ```cd``` command then preprocess the dataset, i.e. see commands below.
```console
user:~$ cd code
user:~$ python3 preprocess.py --dataset ../dataset/spec_df.csv --seed [seed]
```
where ```seed``` is the seed value (```123``` by default). Check if this creates some new directories in the ```datasets/``` directory. <br>

4. Consider one element first (say SiO2). Then run all the models for each of the machine learning algorithms. <i>(See (A) below for details on running non-NN models and (B) below for details on NN models).</i> <br>

5. Then run the Ensemble scripts.
```console
user:~$ python3 ensemble.py --ann [X] --cnn [X] --enets [X] --lasso [X] --linreg [X] --pls [X] --ridge [X] --svr [X] --xgb [X] --element SiO2
```
where ```[X]``` is the output ID result of the specified model. This produces an output in the ```results/``` directory (you do not have to create this directory as this is done automatically when running the models) called ```ensemble_result.txt```, where it gives a summary table of the results -- as well as the ensemble model and its output. <br>

6. (Updated sometime in December 2021): Unfortunately there is a small bug found in the ```ensemble_result.txt```, hence we would have to re-run the Ensemble script. For this, we use the updated Ensemble script, ```ensemble-upd.py```. Create a directory in ```Spectroscopy/spectroscopy-preliminary-results/```. And then create another directory inside this directory called ```SiO2```. Then copy all files from ```Spectroscopy/results/``` to ```Spectroscopy/spectroscopy-preliminary-results/SiO2/``` (you can then safely delete all files in ```Spectroscopy/results/```). <br>

7. Repeat Steps 4-6 for each of the eight elements: ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']. <br>
 
8. Run the updated script as aforementioned in Step 6, i.e. run this command.
```console
user:~$ python3 ensemble-upd.py --element SiO2
```
and you would find a file called ```ensemble_result_upd.txt``` under the ```Spectroscopy/spectroscopy-preliminary-results/SiO2/``` directory. This is now the corrected file. Repeat this step for each of the eight elements: ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']. <br>

<hr>
  
### A. Running the Machine Learning Models (Non-Neural Networks)
This describes how to run the machine learning models, but this is for non-neural network models. These family of models are developed from ```scikitlearn``` libraries. <br>

Firstly, navigate to the ```code/``` directory and then run the script, i.e. run these commands:
```console
user:~$ cd code # no need to run this if you are already in the code/ directory
user:~$ python3 main.py --model [model] --params [path_to_parameters_file] --element [element] --seed [seed]
```
where ```element``` is the element from one of ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O'], ```seed``` is the seed value (```123``` by default), ```model``` is one of the following ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb'], and ```path_to_parameters_file``` is the path to the parameter file (found in the ```params``` directory). <br>

So for example, if you wish to run PLS on SiO2, then you run this command while inside the ```code/``` directory:
```console
user:~$ python3 --model pls --params ../params/pls_params.txt --element SiO2 --seed 123
```
The output will be two files and they are both found in the directory ```results/M```, where ```M``` is a non-NN model found in ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb']:
1. ```outXXX_element=SiO2_model=pls.txt``` (the model results and results of the best-performing parameters for the model)
2. ```outXXX_element=SiO2_out_params=pls.txt``` (the parameters of the best-performing parameters for the model)
where ```XXX``` here is the three-digit output ID of the output result (take note of the ```XXX``` here because this is the one you use for running the ```ensemble.py``` later on). <br>

<hr>

### B. Running Neural Network Models
How to run is not that different. These models (for ANN and CNN) do not rely on ```scikitlearn``` because it is very limited with neural networks and not that efficient. We use ```keras``` here, but to make things faster, we employ ```tensorflow```. <br>

So again navigate to the ```code/``` directory and then run the following:
```console
user:~$ cd code # no need to run this if you are already in the code/ directory
user:~$ python3 main-tf.py --model [model] --params [path_to_parameters_file] --element [element] --seed [seed] --model_id [m_id]
```
which is exactly the same as the one for running the non-neural network machine learning algorithms, except we use ```main-tf.py``` here and that we have an extra parameter. The ```m_id``` here is only relevant for CNNs. You can leave the parameter out for ANN. Two output files will be generated into the ```results/ann/```  directory when we run the ANN model. So it's not different from others. For CNN however, it's a little different (computational cost, memory efficiency issues, etc.). <br>

So, for CNN you need to specify the ```m_id```. Its value is from [1,2,3,4,5]. You need to run the CNN model five times (unlike the other models that you only run once) on a specific element. For the first run ```m_id``` is 1; in the second run, ```m_id``` is 2, and so forth until you reach the fifth run for when ```m_id``` is 5. Make sure that each run is successful (it will a display a success message on your terminal or console), before moving to the next run of the next ```m_id``` value. If for some reason, you made a mistake or there was an error when running say the 3rd run, then you need to go back to ```m_id``` having value 1 (i.e. the first run). So it has to be that CNN runs successfully five consecutive times each with their distinct model IDs from 1 to 5. To verify that this step is done correctly, make sure you have the following files in your ```results/cnn/``` directory generated:
1. ```outXXX_element=E_model=cnn_mod_id=I.txt```
2. ```outXXX_element=E_out_params=cnn_mod_id=I.txt```
where ```XXX``` is the result ID, ```E``` is the element, ```I``` is the ```m_id``` ([1,2,3,4,5]). And there will be 5 pairs of those in that directory (total of those 10 files) for that specific element ```E```. Take note of the result IDs of those 10 files. Take the minimum result ID of those 10, and take note of it and call it ```N``` (which we use when we compile). So for example, the output IDs of the five consecutive successful runs are [28, 29, 30, 31, 32], then ```N=28```. <br>

And then we compile this as follows (only do this for CNN). Still in the ```code/``` directory, run
```console
user:~$ python3 compile.py --out [N] --element [E]
```
where ```[E]``` is the element and ```[N]``` is the ```N``` value obtained earlier. This consolidates the CNN results into a single file, whose output is also going to be generated in the ```results/cnn/```. Note that this compilation cannot be done unless all five CNN runs are consecutively successful. <br>

<hr>

### C. Extras
These things can only be done once all eight steps (see [Usage](https://github.com/techGIAN/Spectroscopy#usage) section). First, naviagate to ```code/```:
```console
user:~$ cd code
```
1. We can see the Actual-vs-Predicted plots of the best models and Ensemble using this command.
```console
user:~$ python3 ensemble-actual-pred-plots.py --element [E]
```
where ```[E]``` is one of the eight elements.

2. We can see the plot for PLS - that determines the appropriate number of components for each of the eight elements
```console
user:~$ python3 pls-plot.py 
```

3. We can see the ternary plots. (TO-DO)
4. We can see the plots of the actual raw data. (TO-DO)

## It Does Not Work...
If it does not work, here are some options:
1. Check the filenames, spelling-wise and sensitive-case-wise.
2. Check the directory you are working in.
3. Check if you have all the Python packages installed.
4. Contact me (last resort), with a detailed description of the error, and which part of the procedure above you are stuck in. I'll try my best to replicate the error and to help.

## Other Things To Do
- [x] Complete the restructured model.
- [ ] Test on another computer.
- [ ] Plot ternary plots
- [ ] Plot actual raw data.
- [x] Plot actual-vs-observed
- [x] Runtime analysis
- [ ] Finish writing the manuscipt/changes.
