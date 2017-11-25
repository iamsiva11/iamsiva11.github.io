---
published: true
title: Graph embeddings 2017 - Part II
layout: post
---
![graph vizualisation in 3d]({{site.baseurl}}/images/3d-graph-viz.png)

This is the second part of the Graph embeddings 2017 blog series. You can find the [first part here](https://iamsiva11.github.io/Graph-embeddings-part-1/)

In the previous blog post we discussed about representation learning and graph embeddings in general, which would serve as the foundation for this blog post as it will go into graph embeddings in much more depth. 

In the early 2000s, researchers developed graph embedding algorithms as part of dimensionality reduction techniques. They  would  construct  a  similarity  graph  for  a  set  of n D-dimensional  points  based  on  neighbourhood  and  then embed  the  nodes  of  the  graph  in  a D-dimensional  vector-space, where d<<D. The idea for embedding was to keep connected  nodes  closer  to  each  other  in  the  vector  space. Laplacian  Eigen maps(LAP)[] and  Locally  Linear  Embedding
(LLE)[] are  examples  of  algorithms  based  on  this  rationale. 

Since 2010, research  on  graph  embedding  has shifted to  obtaining  scalable  graph  embedding  techniques  which leverage the sparsity of real-world networks. For example, Graph Factorisation [] uses an approximate factorisation of the adjacency matrix as the embedding. LINE [] extends this approach and attempts to preserve both first order and second proximities. HOPE [] extends LINE to attempt preserve  high-order  proximity  by  decomposing  the  similarity matrix  rather  than  adjacency  matrix  using  a  generalised Singular  Value  Decomposition  (SVD).  SDNE  []  uses  auto-encoders to embed graph nodes and capture highly non-linear  dependencies.  The  new  scalable  approaches  have  a time complexity of O(|E|).

Recently, on of the pioneering algorithm in graph embedding technique was “DeepWalk” [], followed by LINE [], GraRep [], etc. DeepWalk,	Walklets, LINE (Large-scale Information Network Embedding),	HPE (Heterogeneous Preference Embedding), APP (Asymmetric Proximity Preserving graph embedding), MF (Matrix Factorisation) are some of the important techniques that came up in the recent past. And more importantly, of these methods general  non-linear  models  (e.g.  deep learning based) have shown great promise in capturing the inherent  dynamics  of  the  graph.

Before going to each technique in detail. Lets have a quick overview of what the important pioneering techniques in graph embedding do.

* Laplacian Eigenmaps, Locally Linear Embedding [Early 2000s] - Graph -> adjacency matrix -> latent representation

* HOPE(Higher-Order Proximity preserved Embedding) or (Asymmetric Proximity Preserving graph embedding)APP [KDD 2016] - preserve high-order proximities, capturing the asymmetric transitivity

* DeepWalk [KDD'14] - Node sentences + word2vec

* node2vec [KDD'16] - DeepWalk + more sampling strategies

* SDNE [KDD'16] (Structural Deep Network Embedding) - Deep autoencoder + First-order + second-order proximity

* LINE [Tang’15] (Large-scale Information Network Embedding) - Shallow + first-order + second-order proximity

Other important work worth mentioning:

* Graph Factorization/ MF (Matrix Factorization) (ACM, 2013) 
* GraRep [ACM, 2015] - similar to HOPE
* struc2vec [KDD 2017]
* GraphSAGE [NIPS 2017]
* Graph Convolutional Networks(GCN) [ICLR 2017]
* GENE - Group document + doc2vec(DM, DBOW)

Having seen the various important graph embedding techniques in the recent past. We can group these embedding methods into three broad categories:

* Factorisation based methods
* Random Walk based Methods
* Deep Learning based

## A Taxonomy of Graph Embedding Methods

### Factorisation based methods
---

Matrix factorisation  based  graph  embedding  represent graph property (e.g., node pairwise similarity) in the form of a matrix and factorize this matrix to obtain node embedding. The pioneer studies in graph embedding usually solve graph  embedding  in  this  way.  The  problem  of  graph embedding  can  thus  be  treated  as  a  structure-preserving dimension reduction method which assumes the input data lie  in  a  low  dimensional  manifold.  There  are  two  types of  matrix  factorization  based  graph embedding.  There  are  two  types of  matrix  factorization  based  graph  embedding. One  is  to factorize graph  Laplacian  eigenmaps ,  and  the  other  is  to directly factorize the node proximity matrix. The  matrices  used  to represent  the  connections  include  node  adjacency  matrix, Laplacian  matrix,  node  transition  probability  matrix,  and Katz  similarity  matrix,  among  others. Approaches  to  factorize  the  representative  matrix  vary  based  on  the  matrix properties.  If  the  obtained  matrix  is  positive  semidefinite, e.g.  the  Laplacian  matrix,  one  can  use  eigenvalue  decomposition. For  unstructured  matrices,  one  can  use  gradient descent methods to obtain the embedding in linear time.

Factorization based Methods
* Locally Linear Embedding (LLE)
* Laplacian Eigen maps
* Graph Factorisation (GF) 
* GraRep [Cao’15]
* Higher-Order Proximity preserved Embedding (HOPE/APP) . (aka Asymmetric Proximity Preserving graph embedding

### Random Walk based Methods
---

Random walks have been used to approximate many properties in the graph including node centrality and similarity. They are especially useful when one can either only  partially  observe  the  graph,  or  the  graph  is  too  large to measure in its entirety. Embedding techniques using random walks on graphs to obtain node representations have been proposed: DeepWalk and node2vec are two examples. In random walk based  methods,  the  mixture  of  equivalences  can  be  controlled  to  a  certain  extent  by  varying  the  random  walk parameters. Embeddings learnt  by node2vec with  parameters set  to  prefer  BFS  random  walk  would  cluster  structurally equivalent  nodes  together. On  the  other  hand,  methods which directly preserve k-hop distances between nodes (GF,LE and LLE with k= 1 and HOPE and SDNE with k >1) cluster neighbouring nodes together.

Random Walk based Methods
* DeepWalk
* node2vec

### Deep Learning based
---

The growing research on deep learning has led to a deluge of deep neural networks based methods applied to graphs. Deep auto-encoders have been e.g. used for dimensionality reduction  due to their ability to model non-linear structure in the data. We can interpret the weights of the auto-encoder as a representation of  the  structure  of  the  graph. Recently, SDNE  utilised this ability of deep auto-encoder to generate an embedding model that can capture non-linearity in graphs. As  a  popular  deep  learning model, Convolutional Neural Network (CNN) and its variants  have  been  widely  adopted  in  graph  embedding.

Deep Learning based methods
* SDNE - auto-encoder based(encoder decoder methods)
* GCN - Uses CNN
