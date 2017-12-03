---
published: true
title: Graph embeddings 2017 - Part II
layout: post
---
![graph vizualisation in 3d]({{site.baseurl}}/images/3d-graph-viz.png)

This is the second part of the Graph embeddings 2017 blog series. You can read the [first part here](https://iamsiva11.github.io/Graph-embeddings-part-1/)

In the previous blog post we discussed about representation learning and graph embeddings in general. Which would serve as the foundation for this blog post as this post will go into graph embeddings in much more depth. 

In the early 2000s, researchers developed graph embedding algorithms as part of dimensionality reduction techniques. They would construct  a  similarity  graph  for  a  set  of n D-dimensional  points  based  on  neighbourhood  and  then embed  the  nodes  of  the  graph  in  a D-dimensional  vector-space, where d<<D. The idea for embedding was to keep connected  nodes  closer  to  each  other  in  the  vector  space. Laplacian  Eigen maps(LAP)[1] and  Locally  Linear  Embedding(LLE)[2] are  examples  of  algorithms  based  on  this  rationale. 

Since 2010, research on graph embedding has shifted to obtaining scalable  graph embedding techniques which leverage the sparsity of real-world networks. For example, Graph Factorisation [4] uses an approximate factorisation of the adjacency matrix as the embedding. LINE [3] extends this approach and attempts to preserve both first order and second proximities. HOPE [5] extends LINE to attempt preserve  high-order  proximity  by  decomposing  the  similarity matrix rather than adjacency matrix using a generalised Singular Value Decomposition(SVD). SDNE[6] uses auto-encoders to embed graph nodes and capture highly non-linear dependencies. The new scalable approaches have a time complexity of O(|E|).

Recently, on of the pioneering algorithm in graph embedding technique was “DeepWalk” [8], followed by LINE[3], GraRep[7], etc. DeepWalk,	Walklets, LINE (Large-scale Information Network Embedding),	HPE(Heterogeneous Preference Embedding), APP(Asymmetric Proximity Preserving graph embedding), MF(Matrix Factorisation) are some of the important techniques that came up in the recent past. And more importantly, of these methods general  non-linear  models(e.g.  deep learning based) have shown great promise in capturing the inherent  dynamics of the graph.

We can group these embedding methods into three broad categories. And explain the characteristics of each of these categories and provide a summary of a few representative approaches for each category:

* Factorisation based methods
* Random Walk based Methods
* Deep Learning based

## Classification of Graph Embedding Methods

### Factorisation based methods
---

Matrix factorisation based graph embedding represent graph property (e.g., node pairwise similarity) in the form of a matrix and factorize this matrix to obtain node embedding. The pioneer studies in graph embedding usually solve graph embedding in this way. The problem of graph embedding can thus be treated as a structure-preserving dimension reduction method which assumes the input data lie in a low dimensional manifold. There are two types of matrix  factorization based graph embedding. One is to factorize graph Laplacian eigenmaps, and the other is to directly factorize the node proximity matrix. 

The matrices used to represent the connections include node adjacency matrix, Laplacian matrix, node transition probability matrix, and Katz  similarity matrix,  among  others. 

Approaches  to  factorize  the  representative  matrix  vary  based  on  the  matrix properties.  If  the  obtained  matrix  is  positive  semidefinite, e.g.  the  Laplacian  matrix,  one  can  use  eigenvalue  decomposition. For  unstructured  matrices,  one  can  use  gradient descent methods to obtain the embedding in linear time.

Some of the important Factorization based Methods are mentioned below:

* Locally Linear Embedding (LLE)
* Laplacian Eigen Maps(LAP)
* Graph Factorisation (GF) 
* GraRep [Cao’15]
* Higher-Order Proximity preserved Embedding (HOPE/APP) . (aka Asymmetric Proximity Preserving graph embedding

### Random Walk based Methods
---

Random walks have been used to approximate many properties in the graph including node centrality and similarity. They are especially useful when one can either only  partially  observe  the  graph,  or  the  graph  is  too  large to measure in its entirety.

Embedding techniques using random walks on graphs to obtain node representations have been proposed: DeepWalk and node2vec are two examples. 

In random walk based  methods,  the  mixture  of  equivalences  can  be  controlled  to  a  certain  extent  by  varying  the  random  walk parameters. Embeddings learnt  by node2vec with  parameters set  to  prefer  BFS  random  walk  would  cluster  structurally equivalent  nodes  together. On  the  other  hand,  methods which directly preserve k-hop distances between nodes (GF,LE and LLE with k= 1 and HOPE and SDNE with k >1) cluster neighbouring nodes together.

Random Walk based Methods:

* DeepWalk
* node2vec

### Deep Learning based
---

The growing research on deep learning has led to a deluge of deep neural networks based methods applied to graphs. Deep auto-encoders have been e.g. used for dimensionality reduction  due to their ability to model non-linear structure in the data. We can interpret the weights of the auto-encoder as a representation of  the  structure  of  the  graph. Recently, SDNE  utilised this ability of deep auto-encoder to generate an embedding model that can capture non-linearity in graphs. 

As  a  popular  deep  learning model, Convolutional Neural Network (CNN) and its variants  have  been  widely  adopted  in  graph  embedding.

Deep Learning based methods:

* SDNE - auto-encoder based(encoder decoder methods)
* GCN - Uses CNN

Having seen the taxonomy of approaches in the Graph embeddigns technique; lets have a quick overview of what the important pioneering techniques in graph embedding do. And provide a context of the research developemnt and progress of in the Graph embeddings space:

* Laplacian Eigenmaps, Locally Linear Embedding [Early 2000s]: 
Graph -> adjacency matrix -> latent representation (Factorisation based methods)

* HOPE(Higher-Order Proximity preserved Embedding) or (Asymmetric Proximity Preserving graph embedding)APP [KDD 2016] - preserve high-order proximities, capturing the asymmetric transitivity. (Factorisation based methods)

* DeepWalk [KDD'14] - Basic idea: apply word2vec to non-NLP data. Random walk distance is known to be good features for many problems (i.e. Node sentences + word2vec) (Randomwalk based methods)

* node2vec [KDD'16] - DeepWalk + more sampling strategies (Randomwalk based methods)

* SDNE [KDD'16] (Structural Deep Network Embedding): Use  deep  autoencoders  to preserve  the  first  and  second  order  network proximities(i.e. Deep autoencoder + First-order + second-order proximity). They achieve this by jointly optimizing the two proximities. (Deep-learning based methods)

* LINE [Tang’15] (Large-scale Information Network Embedding) - Shallow + first-order + second-order proximity (Other than the 3 categories mentioned above)

Other important techniques worth mentioning:

* Graph Factorization/ MF (Matrix Factorization) [ACM, 2013]
* GraRep [ACM, 2015] - similar to HOPE
* struc2vec [KDD 2017]
* GraphSAGE [NIPS 2017]
* Graph Convolutional Networks(GCN) [ICLR 2017]

# REFERENCES

[1] M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral techniques for embedding and clustering,” in NIPS, vol. 14, no. 14, 2001, pp. 585–591.

[2] S. T. Roweis and L. K. Saul, “Nonlinear dimensionality reduction by locally linear embedding,” Science, vol. 290, no. 5500, pp. 2323–2326, 2000.

[3] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei, “Line: Large-scale information network embedding,” in Proceedings 24th International Conference on World Wide Web, 2015, pp. 1067–1077.

[4] A. Ahmed, N. Shervashidze, S. Narayanamurthy, V. Josifovski,
and A. J. Smola, “Distributed large-scale natural graph factorization,”
in Proceedings of the 22nd inter

[5] M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu, “Asymmetric transitivity preserving graph embedding,” in Proc. of ACM SIGKDD, 2016, pp. 1105–1114.

[6] D. Wang, P. Cui, and W. Zhu, “Structural deep network embedding,”
in Proceedings of the 22nd International Conference on
Knowledge Discovery and Data Mining. ACM, 2016, pp. 1225–1234.

[7] S. Cao, W. Lu, and Q. Xu, “Grarep: Learning graph representations
with global structural information,” in Proceedings of the 24th ACM
International on Conference on Information and Knowledge Management.
ACM, 2015, pp. 891–900.

[8] B. Perozzi, R. Al-Rfou, and S. Skiena, “Deepwalk: Online learning
of social representations,” in Proceedings 20th international conference
on Knowledge discovery and data mining, 2014, pp. 701–710.