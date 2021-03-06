# transboost -
This repository contains an implementation of the TransBoost algorithm, a transfer learning algorithm based on boosting of weak projections from the target domain to the source domain. There are two versions : TransBoost3 requires python3 and has additional features, using tensorflow (which seems to bring up issues in python2). TransBoost2 can be run both by python2 and python3.

*Authors* : Antoine Cornuéjols, Raphaël Olivier, Sema Akkoyunlu, Pierre-Alexandre Murena

## Usage
### Requirements
#### in TransBoost2
* python 2.7.13
* matplotlib 2.0.2
* numpy 1.12.1
* pandas 0.20.1
* scikit-learn 0.18.1
* scipy 0.19.0
#### in TransBoost3
* python 3.5.3
* matplotlib2.0.2
* numpy 1.12.1
* pandas 0.20.2
* scikit-learn 0.18.1
* scipy 0.19.0
* tensorflow 1.1.0

### Apply TransBoost
To apply TransBoost with this implementation, you will need the *transBoost* package. What you have to do is :
* Build or gather your labelled target data (training and testing).
* Set a number of boosting steps.
* Create a TransBoost object and feed it with your data and number of steps. (see *transBoost.TransBoost*).
* Choose a projection search mode and create a ProjFinder object (see *projections.ProjFinder*).
* Build or define your source hypothesis (function in stupid/random mode, graph in neural mode) and feed it to your ProjFinder.
* Define your projection space (in stupid/random mode define functions and their parameters sets to explore ; in neural mode (TransBoost3) define the properties of your projection graph) and feed it to your projFinder.
* Feed the ProjFinder to the TransBoost object.
* __learn__ (*TransBoost.learn*) and __test__ (*TransBoost.test*).

Take a look at the *tools* package to help you with the preliminary steps.

### Execute the code
You should run your python scripts from the main folder (TransBoost2 or TransBoost3) for the imports to work smoothly. 
For example, in this actual code the only executable file is *main.py*, which imports and calls one of the example files. Alternatively, you can add this folder to your PYTHONPATH.

## Detailed description of TransBoost2
This version contains 4 packages :
* *transBoost* : the core package that contains most of the actual algorithm.
* *tools* : auxiliary functions, used in the classes and functions in transBoost and useful for developpers that want to use transBoost.
* *series* : material relative to one specific TransBoost application : classification of incomplete time series. This code has been used to test the algorithm and may give some insight about the way to use transBoost material.
* *examples* : ready-to-execute functions showing some ways to use TransBoost. In this version they are based on the time series application.

There is also a *main.py* file, which is here the only executable file. It calls one of the example functions.
### transBoost
In transboost, the user provides a source hypothesis, a space of projections from target to source, and target data. The algorithm will find several weak projections and combine them with an AdaBoot-like method, where weak classifiers on target domain are given by weak projections composed with the source hypothesis.
#### boosting.py
It is the file that does most of the work, appart from searching and applying projections (which is actually the real big part). There are four functions here :
* *boosting* : it is the "Training" function. Given data, labels, a number of boosting steps and a structure to find projections (the ProjFinder - see below), it will for each step compute each projection's coefficient, update examples's weights (like in AdaBoost). It returns a *projections* object returned by projFinder (a structure that contains projections), and lists containing each projection's error rate, coefficient and research time (for log purposes).
* *computeWeights* : an auxiliary function used by boosting to compute each point's weight at every step.
* *test* : it is the "Testing" function. Given data, its correct labels, a structure that contains selected projections, and each projection's coefficient, it will compute the transBoost prediction. It returns the prediced labels, the error rate and the error of each weak classifier.
* *run* : it is the prediction function on unlabelled data. Given data, a structure that contains selected projections, and each projection's coefficient, it will compute and return the transBoost prediction.
#### predictions.py
In a transBoost application there will be two classes needed to deal with projections : one to search and find projections, one to contain and apply them. This particular structure was used because there are many ways to search projections ; and although a projection is simply a function from a target space to a source space, different ways of representing them may be used depending on the chosen research method. Therefore some high-level abstraction was needed to properly deal with projections in general cases.

The **ProjFinder** class deals with projection research. It is initiated with a mode that decides how projections will be searched for. Depending on the mode, some attributes may not be used. In TransBoost2 there are only two modes right now (one more in TransBoost3) :
* "*stupid*" mode corresponds to ordered exhaustive research.
* "*random*" mode correspond to a totally random research.

In both modes, we set a source hypothesis (function that classifies a dataset of source points) and we consider that a projection is a function + a set of parameters. In projFinder the projection space will be a dictionnary where keys are functions (depending on parameters) and values are collections of parameters. For each function, parameters are all tried (repsctively in order or randomly) until we find one with tolerable weighted error (that is, below a threshold attribute). If no one is found we just keep the best one. And if the best one is not better than random choice (that is, above a randombar attribute slightly below 0.5) there is no need to run more boosting steps.

In random mode an additional timelimit attribute is used : after it is crossed in searching parameters for one projection function, we stop and move to the next function.

A **projFinder** can be initialised with the *init* method, which takes the data. Research is done by calling *search* which takes the data weights as parameters: it calls a specific searching method depending on the mode.


We also need a class to store a list of projections, apply projections and compose them with a hypothesis. It is used by projFinder and returned by it to the *boosting.boosting* function. It is called by *boosting.test* to compute predicted labels. Its methods are :
* The constructor that takes the source hypothesis
* *proj* : apply a given projection, one to add a projection to the list
* *add* : add a projection to the list.
* *keepLast* : delete projections appart from the last one (useful if the last one has 100% accuracy)
* *labelslist* : takes data as parameters, apply the projections to it, compose with the source hypothesis and returns the labels.
* *printProjections* : for logging purposes
    
The **Projections** class does all this for projections in the format used in stupid/random research. Other classes may be used as long as they have the methods above. Another is defined in TransBoost3 for neural mode. **Projections**'s attributes are the source hypothesis (function that returns labels of a source dataset), a list of projection functions and the corresponding list of parameters.
#### transBoost.py
This file defines the **TransBoost** class. It does not hold much computation, but contains all required parameters and data to run TransBoost, encapsulates calls to *boosting* functions and holds the logs and results of any training or testing phase. Once a **ProjFinder** is set up, it is the only class that should be called in a TransBoost application.

Its attibutes are training data, testing data, the number of boosting steps, a ProjFinder object, a projections object, a *alphas* list that should contain coefficients of selected projections and a *log* list that contains all logs.

Its methods are :
* A no-parameters constructor.
* Attribute-reading and writing methods.
* Controle methods used to check parameters are set.
* Log methods used to print, return and clear the previous logs.
* *learn* : the "training" function. Calls *boosting.boosting* on training data, saves alphas, projections and logs. Also calls *boosting.test* on training data, to compute and return training error.
* *test* : the "testing" function. Calls *boosting.test* on testing data, saves logs and returns the testing error.
* *run* : the prediction function. Calls *boosting.run* on given data and returns the predicted labels.
### tools
#### data.py
Generic functions that manipulates datasets and files. Datasets are basically always numpy arrays (one for the features, one for the labels).
* *importCSV* : takes a path and the csv delimiter, imports a csv into two numpy arrays X (features) and y (labels). The csv must hold labels in its first column.
* *exportCSV* : takes two numpy arrays a path and a delimiter and exports the datasets in csv format at the path, with labels in its first column.
* *randomShuffle* : takes a dataset, labels, and a tuple *prop* of positive floats whose sum should not exceed 1. Returns a tuple of disjoint samples of the dataset, with fractions of its elements equal to the elements of *prop*.
#### display.py
Functions used to print collections. They have an optional argument *logFile* containing a path. If filled, the collection will be printed and saved in the file. If not, it will be printed on standard output. They also have an optional argument *title*, a String that could be used as a header in the document. There is *displayDict* for dictionnaries and *displayList* for lists.
#### learning.py
Generic learning functions, useful to build hypothesis or compute errors.
* *error* : given correct and predicted data, it computes the prediction error, that is the proportion of wrong predictions.
* *weightedError* : the same except each example is assigned a weight and counts as much in the error rate. An optional rgument D holds weights ; if not filled uniform weights will be used (that is, usual error).
* *testhyp* : given a hypothesis, features, labels and weights, classifies the data and returns the prediction and the weighted error.
* *learnSVM* : takes training data X and labels y. Returns a rbf kernel SVM fitted with the given data, along with the training error. An argument gamma may take the coefficient for svm (see sklearn.svm.SVC).
* *testSVM* : takes a trained kernel, features, labels and weights. Returns weighted error and prediction.
### series
Time series are a particular application of transfer lerning, where the source domain contains time series, and the target domain contains similar time series, but shorter. It may be very useful to learn how to classify short, incomplete time series, while your labelled dataset are mostly composed of complete series ; hence the idea to project incomplete series on complete ones using TransBoost.

Our datasets contain long series. To classify short ones, we will cut them.

It is designed to us random mode in the **ProjFinder** class.
#### series.py
Functions that help building hypothesis on time series. One is *svmhyp* : given data and labels, it returns a svm-based hypothesis using *learning.learnSVM*.

The others are projection function that help filling the projFinder function. For each function *f*, there is a *f_param* function that, given some information (typically the lengths of source and target series), will return a collection (or yield a generator) of parameters sets. Look at each one for some details.
#### seriesGeneration.py
Functions that help create of manipulate series datasets.
* *genere_dataset* : given a number of elements, a length of series, a path and various sets parameters (ranges of oscillations, deviations of noise, slopes of main classes,...), builds and returns a dataset.
* *cutSeries* : given a dataset of series *X* of length *L* and a length *l* < *L*, returns the dataset whose elements are the l first measures of the elements of X.
#### seriesTesting
Functions used to launch large sets of experiences to apply transBoost on time series and save and display the results.
* *testseries* : the main function. It takes lists of parameters (length of source series, length of target series, number of boosting steps) and dataset paths, along with a result file path and a number *n*. For each possible combinaison of parameters and datasets, a source hypothesis is built and *n* experiences are run. An experience is :
    * Select training and testing sample of cut series.
    * Train and test a naive target classifier (a rbf svm on target domain is harcoded).
    * Train and test TransBoost.
    
Then for each error mean and standard deviation on the *n* experiences are computed, and stored as csv in the result file.
* *singletest* : an auxiliary function used by *testseries* to run transBoost once. It defines and initializes **TransBoost** and **ProjFinder** objects, runs the algorithm and returns training and testing errors. It also prints logs in shell.
* svmreg : an auxiliary function, that can be used instead of target svm as a naive classifier. It computes regression on incomplete series to naively project them on source domain and apply the source hypothesis. The function returns training and testing errors.
* *simplifyCSV* : testseries generates csv with too precise floats to read. This function takes a csv path and simplifies it to two significant figures after comma.
* *displayExperience* : takes a pandas dataFrame (e.g. read from a csv) and two column names. It plots a graph of points, one column being the x values and the other the y values. It can be used to show how TransBoost performs compared to the naive target classifier.
### examples
Each file here has its code in a *run* function. It is called in the *main.py* file.
#### series0.py
Generates and saves the datasets used in the other series examples.
#### series1.py
A simple use of TransBoost on one dataset.
#### series2.py
The set of experiences we used to test transBoost. It mainly calls *series.seriesGeneration.testseries* with quite a large number of parameters.
#### series3.py
A smaller, computable set of experiences to show how *series.seriesGeneration.testseries* works.
## Differences in TransBoost3
TODO
