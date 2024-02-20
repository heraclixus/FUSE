# FUSE

## Introduction to FUSE 

Fuzzy Set Embedding (FUSE) is a general representation learning framework for __Fuzzy sets__, which are generalizsation of crisp sets and can be used to model semantic concepts or objects with an inherent characterization of __volume__ and __ambiguity__. We divide the project into several stages, and in each stage we target a different application of FUSE to compare against traditional vector and geometry(box)-based embeddings; we also develop the theory behind the measure-theoretic approximation of fuzzy set from a general non-parametric simple approximation to a parametric distribution based approximation (Gumbel distribution). 

## Repository Structure 

under the `src` directory, we include codes for different applications:
- Taxonomy Expansion, under the `taxonomy` subdirectory.
- Knowledge Graph Embedding, under the `knowledge_graph` subdirectory.
- Language Modeling, Word Embedding, under the `langauge_modeling` subdirectory.

### Stage One: Taxonomy Expansion

This part corresponds to our ACL submission: _FUSE: Measure-Theoretic Fuzzy Set Embedding for Taxonomy Expansion_. The dataset in this case follows from [Box-Taxo](https://github.com/songjiang0909/BoxTaxo), and the original data can be obtained from SemEval-2016 Task 13: Taxonomy Extraction Evaluation. Processed data is also under `data` folder, obtained from [STEAM](https://github.com/yueyu1030/STEAM).

### Stage Two: Knowledge Graph Embedding 

For knowledge graph, we base partly the code from [Fuzzy-QE](https://github.com/stasl0217/FuzzQE-code) and [BetaE](https://github.com/snap-stanford/KGReasoning). The dataset should be downloaded from [here](http://snap.stanford.edu/betae/KG_data.zip) and put it under `data` folder. We are still updating this folder to add more baseline models and to improve existing model. 

The directory structure should be like `[PROJECT_DIR]/data/NELL-betae/train-queries.pkl.`


### Stage Three: Language Modeling, Word Embedding

This part corresponds to our planned submission to Neurips, _Measure-Theoretic Representation of Fuzzy Sets_, which covers a more in-depth experimental and theoretical development of FUSE for various applications in neuro-symbolic reasoning. This part is currently under construction. 

