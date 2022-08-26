# Spectroscopy
Machine Learning Project where we compare machine learning models' performances evaluated on spectrum datasets and then constructing ensembles

## Abstract
The exploration and analyses of chemical components in (extra-)terrestrial geological materials (such as asteroids and meteorites) are insightful in modern research. Laser-Induced Breakdown Spectroscopy (LIBS) is a popular method for analyzing the chemical attributes of geologic samples -- which scientists use to study and understand planetary bodies and their complex histories. In the literature, several machine learning models that produce high-accuracy predictions have been proposed. In our work, we compared the performances of such models in predicting elemental abundances on a certain spectroscopic dataset. Models included Partial Least Squares (PLS), Extreme Gradient Boost Machines (XGB), Neural Networks, and Linear Models. In our results, we showed how PLS and XGB are superior in terms of high predictive power, their ability to generalize, and their reasonably efficient runtimes. In addition, we proposed Ensemble models that aggregates predictions of top-tier models, and observed that they can be desirable. We intend to gain better understanding of how these models perform in predicting elemental compositions on specific spectrum (LIBS) datasets.

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

## Citation
If you find our work useful in your research, please kindly cite us. Thank you.

<b>Plain text:</b>

<blockquote>
Alix, G, Lymer, E, Zhang, G, Daly, M, Gao, X. A comparative performance of machine learning algorithms on laser-induced breakdown spectroscopy data of minerals. <i>Journal of Chemometrics</i>. 2022; e3400. doi: <a href="https://doi.org/10.1002/cem.3400">10.1002/cem.3400</a>
</blockquote>

<b>BibTeX:</b>
```bibtex
@article{alix2022mllibs,
   author = {Alix, Gian and Lymer, Elizabeth and Zhang, Guanlin and Daly, Michael and Gao, Xin},
   title = {A comparative performance of machine learning algorithms on laser-induced breakdown spectroscopy data of minerals},
   journal = {Journal of Chemometrics},
   pages = {e3400},
   keywords = {chemometrics, geochemistry, LiBS, machine learning models},
   doi = {https://doi.org/10.1002/cem.3400},
   url = {https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3400},
   eprint = {https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.3400}
}
```




<i> Please contact us for any bugs, thank you. </i>
