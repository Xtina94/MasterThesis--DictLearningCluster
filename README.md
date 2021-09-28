# DictLearningCluster

The project intends to learn a graph dictionary knowing the graph signal structure and assuming that the graph signal is smooth. Results are then analyzed to check whether 
the smoothness constraint helps in learning the graph  dictionary.
For insights on what a graph disctionary is, have a look at https://arxiv.org/abs/1401.0887.

- At https://github.com/Xtina94/DictLearningCluster/tree/master/GeneratingKernels there are some scripts generating the test polynomial functions used to produce a graph signal.
  These functions are called kernels
- At https://github.com/Xtina94/DictLearningCluster/tree/master/DictionaryLearning/AlphaStructure one version of the learning algorithm includes the signal smoothness prior in
  the objective function of the optimization problem. This is done through adding a series of unknow parameters (under the letter alpha) to the objective polynomial function.
  https://github.com/Xtina94/DictLearningCluster/blob/master/DictionaryLearning/AlphaStructure/Polynomial_Dictionary_Learning.m is the main file
- At https://github.com/Xtina94/DictLearningCluster/tree/master/DictionaryLearning/Constraints the other version of the learning algorithm includes the signal smoothness prior in the optimization problem constraints.
  https://github.com/Xtina94/DictLearningCluster/blob/master/DictionaryLearning/Constraints/Polynomial_Dictionary_Learning.m is the main file
