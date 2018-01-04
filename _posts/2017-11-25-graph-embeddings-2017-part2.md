---
published: true
title: Graph embeddings 2017 - Part II
layout: post
comments: true
---
![graph vizualisation in 3d]({{site.baseurl}}/images/3d-graph-viz.png)


## Table  of Contents
 
* Introduction
* History annd progress of Graph embeddings
* Classification of Graph Embedding Methods
	* Factorisation based methods
	* Random Walk based Methods
	* Deep Learning based
* A brief summary of pioneering graph embedding techniques      
* Conclusion
* Advanced resources for further learning

---

# Introduction

In the previous blog post we discussed about **_representation learning and graph embeddings_** in general. Which would serve as the foundation for this blog post, as this post will go into graph embeddings in much more depth. 

This is the second part of the Graph embeddings 2017 blog series. You can read the [first part here](https://iamsiva11.github.io/Graph-embeddings-part-1/)

---

# History annd progress of Graph embeddings

In the _**early 2000s**_, researchers developed graph embedding algorithms as part of _dimensionality reduction_ techniques. They would construct  a  _similarity  graph_  for  a  set  of n _D-dimensional_  points  based  on  **neighbourhood**  and  then embed  the  nodes  of  the  graph  in  a D-dimensional  vector-space, where _d<<D_. The idea for embedding was to keep connected  nodes  closer  to  each  other  in  the  vector  space. **Laplacian  Eigen maps(LAP)**[1] and **Locally  Linear  Embedding(LLE)**[2] are  examples  of  algorithms  based  on  this  rationale. 

_**Since 2010**_, research on graph embedding has shifted to obtaining scalable graph embedding techniques which leverage the sparsity of real-world networks. For example, **Graph Factorisation**[4] uses an approximate factorisation of the adjacency matrix as the embedding. **LINE(Large-scale Information Network Embedding)**[3] extends this approach and attempts to preserve both first order and second proximities. **HOPE**[5] extends LINE to attempt preserve  high-order  proximity  by  decomposing  the  similarity matrix rather than adjacency matrix using a generalised Singular Value Decomposition(SVD). **SDNE(Structural deep network embedding)**[6] uses auto-encoders to embed graph nodes and capture highly non-linear dependencies.

_Recently_, on of the **pioneering algorithm** in graph embedding technique was “**DeepWalk**” [8], followed by **LINE**[3], **GraRep**[7], etc. _DeepWalk, Walklets, LINE[3], HPE(Heterogeneous Preference Embedding), APP(Asymmetric Proximity Preserving graph embedding), MF(Matrix Factorisation)_ are some of the important techniques that came up in the recent past. And more importantly, of these methods general  _non-linear  models(e.g.  deep learning based)_ have shown great promise in capturing the inherent  dynamics of the graph.

## Classification of Graph Embedding Methods

We can group these embedding methods into **three broad categories**. And explain the characteristics of each of these categories and provide a summary of a few representative approaches for each category:

* **Factorisation based methods**
* **Random Walk based methods**
* **Deep Learning based methods**

### Factorisation based methods
---

**Matrix factorisation** based graph embedding represent graph property (e.g., node pairwise similarity) in the form of a matrix and _factorize this matrix_ to obtain node embedding. The pioneer studies in graph embedding usually solve graph embedding in this way. The problem of graph embedding can thus be treated as a structure-preserving dimension reduction method which assumes the input data lie in a low dimensional manifold. There are two types of matrix  factorization based graph embedding. _One is to factorize graph Laplacian eigenmaps, and the other is to directly factorize the node proximity matrix_.

The matrices used to represent the connections include _node adjacency matrix, Laplacian matrix, node transition probability matrix, and Katz similarity matrix_,  among  others. 

Approaches  to  factorize  the  representative  matrix  vary  based  on  the  matrix properties.  If  the  obtained  matrix  is  positive  semidefinite, e.g.  the  Laplacian  matrix, one  can  use  _**eigenvalue  decomposition**_. For  unstructured  matrices,  one  can  use  _**gradient descent methods**_ to obtain the embedding in linear time.

Some of the important Factorization based Methods are mentioned below:

* **Locally Linear Embedding (LLE)**
* **Laplacian Eigen Maps(LAP)**
* **Graph Factorisation (GF)** 
* **GraRep [Cao’15]**
* **Higher-Order Proximity preserved Embedding (HOPE). (aka Asymmetric Proximity Preserving graph embedding(APP))**

## Random Walk based Methods
---

Random walks have been used to approximate many properties in the graph including node centrality and similarity. They are especially useful when one can either only  partially  observe  the  graph,  or  the  graph  is  too  large to measure in its entirety.

Embedding techniques using random walks on graphs to obtain node representations have been proposed: **DeepWalk** and **node2vec** are two examples. 

In random walk based  methods,  the  mixture  of  equivalences  can  be  controlled  to  a  certain  extent  by  varying  the  random  walk parameters. Embeddings learnt  by node2vec with  parameters set  to  prefer  BFS  random  walk  would  cluster  **structurally equivalent**  nodes  together. 

On  the  other  hand,  methods which directly preserve k-hop distances between nodes (GF,LE and LLE with k=1 and HOPE and SDNE with k >1) cluster neighbouring nodes together.

Random Walk based Methods:

* **DeepWalk**
* **node2vec**

## Deep Learning based
---

The growing research on deep learning has led to a deluge of deep neural networks based methods applied to graphs. Deep **auto-encoders** have been e.g. used for dimensionality reduction  due to their ability to model non-linear structure in the data. We can interpret the weights of the auto-encoder as a representation of  the  structure  of  the  graph. Recently, SDNE utilised this ability of deep auto-encoder to generate an embedding model that can capture non-linearity in graphs. 

As  a  popular  deep  learning model, Convolutional Neural Network (CNN) and its variants  have  been  widely  adopted  in  graph  embedding. Recent one being, **Graph Convolutional Networks**(GCN)[9] [ICLR 2017].

Deep Learning based methods:

* **SDNE** - auto-encoder based(encoder decoder methods).
* **GCN** - CNN based.

# A brief summary of pioneering graph embedding techniques

Having seen the taxonomy of approaches in the Graph embeddings technique; lets have a _quick overview of what the important pioneering techniques_ in graph embedding do. And provide a context of the research developemnt and progress of in the Graph embeddings space:

* **Laplacian Eigenmaps, Locally Linear Embedding** [Early 2000s] : Graph -> adjacency matrix -> latent representation (Belongs to Factorisation based methods)

* **HOPE**(Higher-Order Proximity preserved Embedding) [KDD 2016] :  preserve high-order proximities, capturing the asymmetric transitivity. (Belongs to Factorisation based methods). aka Asymmetric Proximity Preserving graph embedding(APP).

* **DeepWalk** : [KDD'14] : Basic idea: apply word2vec to non-NLP data. Random walk distance is known to be good features for many problems (i.e. Node sentences + word2vec) (Belongs to Randomwalk based methods)

* **node2vec** [KDD'16] : DeepWalk + more sampling strategies (Belongs to Randomwalk based methods)

* **SDNE** [KDD'16] : (Structural Deep Network Embedding): Use  deep  autoencoders  to preserve  the  first  and  second  order  network proximities(i.e. Deep autoencoder + First-order + second-order proximity). They achieve this by jointly optimizing the two proximities. (Deep-learning based methods)

* **LINE** [Tang’15] : (Large-scale Information Network Embedding) - Shallow + first-order + second-order proximity (Belongs to none of the 3 broad categories mentioned above)

Other important techniques worth mentioning:

* **Graph Factorization/ MF (Matrix Factorization)** [ACM, 2013]
* **GraRep** [ACM, 2015] - similar to HOPE
* **struc2vec** [KDD 2017]
* **GraphSAGE** [NIPS 2017]


# Conclusion

This post was meant to provide a big-picture for learning and getting started with graph embeddings. In the upcoming posts, I'll highlight some of the current trends and future directions. While going through each important technique, applied part, performance comparision of these techniques, and some practical advice while working with graph embeddings in detail.

I've undoubtedly failed to mention many other areas that are equally important and noteworthy. Please let me know in the comments below what I missed, where I made a mistake or misrepresented a method, or just which aspect of graph embeddings you find particularly exciting or unexplored.


## Advanced resources for further learning in **Graph Embeddings**

[Network Embedding Survey and Resources](https://github.com/nate-russell/Network-Embedding-Resources)

[A curated list of network embedding techniques.](https://github.com/chihming/awesome-network-embedding)

[Must-read papers on network representation learning (NRL)/network embedding (NE)](https://github.com/thunlp/NRLPapers)


# REFERENCES

[1] [M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral techniques for embedding and clustering”](http://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NIPS_01.pdf)

[2] [S. T. Roweis and L. K. Saul, “Nonlinear dimensionality reduction by locally linear embedding”](http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf)

[3] [J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei, “Line: Large-scale information network embedding”](http://dl.acm.org/citation.cfm?id=2741093)

[4] [A. Ahmed, N. Shervashidze, S. Narayanamurthy, V. Josifovski,
and A. J. Smola, “Distributed large-scale natural graph factorization”](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf)

[5] [M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu, “Asymmetric transitivity preserving graph embedding”](http://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)

[6] [D. Wang, P. Cui, and W. Zhu, “Structural deep network embedding”](https://dl.acm.org/citation.cfm?id=2939753)

[7] [S. Cao, W. Lu, and Q. Xu, “Grarep: Learning graph representations
with global structural information”](http://dl.acm.org/citation.cfm?id=2806512)

[8] [B. Perozzi, R. Al-Rfou, and S. Skiena, “Deepwalk: Online learning
of social representations"](https://arxiv.org/abs/1403.6652)

[9] [Graph Convolutional Networks (GCNs)](https://tkipf.github.io/graph-convolutional-networks/)

[10] [Network Representation Learning: A Revisit in the Big Data Era](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/Network%20Representation-Tutorial.pdf)

[11] [Part III: Network Embedding with Deep Learning, Renmin University of China](http://www.cips-cl.org/static/CCL2016/tutorialpdf/T3A_%E7%A4%BE%E4%BC%9A%E8%AE%A1%E7%AE%97_part3.pdf)

[12] [Convolutional Neural Networks on Graphs](http://www.ntu.edu.sg/home/xbresson/pdf/talk_xbresson_gcnn.pdf)

[13] [Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://nlp.csai.tsinghua.edu.cn/%7Elzy/publications/aaai2015_transr.pdf)

[14] [Aditya Grover, node2vec: Scalable Feature Learning for Networks](https://dl.acm.org/citation.cfm?id=2939672.2939754)
