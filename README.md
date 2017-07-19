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
* *boosting* : it is the "Training" function. Given data, labels, a number of boosting steps and a structure to find projections (the ProjFinder - see below), it will for each step compute each projection's coefficient, update examples's weights (like in AdaBoost). It returns the projections that projFinder found, and lists containing each projection's error rate, coefficient and research time (for log purposes).
* *computeWeights* : an auxiliary function used by boosting to compute each point's weight at every step.
* *test* : it is the "Testing" function. Given data, its correct labels, a structure that contains selected projections, and each projection's coefficient, it will compute the transBoost prediction. It returns the prediced labels, the error rate and the error of each weak classifier.
* *run* : it is the prediction function on unlabelled data. Given data, a structure that contains selected projections, and each projection's coefficient, it will compute and return the transBoost prediction.


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
