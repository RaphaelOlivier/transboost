# transboost -
This repository contains an implementation of the TransBoost algorithm, a transfer learning algorithm based on boosting of weak projections from the target domain to the source domain. There are two versions : TransBoost3 requires python3 and has additional features. TransBoost2 can be run both by python2 and python3.

*Authors* : Antoine Cornuéjols, Raphaël Olivier, Sema Akkoyunlu, Pierre-Alexandre Murena
## Description of TransBoost2
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
### series
### examples
## Differences in TransBoost3
TODO
## Usage
### Requirements
#### in TransBoost2
#### in TransBoost3
### Execute the code
### Apply TransBoost
