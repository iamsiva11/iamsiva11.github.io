---
published: false
---
## A novel approach to feed and train extra features in Seq2seq (Tensorflow & Pytorch)

---
---

## Table  of Contents
 
* Intro
* So what’s the problem?
* Sequence modelling before deep learning 
* Data input to CRF models
* Technique
	* Pseudocode and illustration
	* Pytorch code
	* Tf code 
* Further notes on the experiment 
* Conclusion

---
---

# Introduction

Recent developments in deep learning have given rise to a powerful alternative – discriminative models called sequence-to-sequence  models, can  be  trained  to  model  the  conditional probability distribution of the output transcript sequence given the input sequence, directly without inverting a generative  model.
 
Encoder-decoder neural models (Sutskever et al., 2014) are a generic deep-learning approach to sequence-to-sequence translation (Seq2Seq) tasks. Encoder Decoder network, is a model consisting of two separate RNNs called the encoder and decoder. The encoder reads an input sequence one item at a time, and outputs a vector at each step. The final output of the encoder is kept as the context vector. The decoder uses this context vector to produce a sequence of outputs one step at a time. The performance  of  the  original  sequence-to-sequence  model  has equal Contribution been  greatly  improved  by  the  invention  of soft attention,  which  made  it  possible  for  sequence-to sequence  models  to  generalize  better  and  achieve  excellent  results  using much  smaller  networks  on  long  sequences. The  sequence-to-sequence  model  with  attention  had  considerable  empirical success on machine translation, speech recognition, image caption generation, and question answering.

If you want to quickly go through the progress of seq2seq research. Paper notes of the foundational papers in seq2seq literature are available [here](https://github.com/iamsiva11/DLNLP-papernotes/tree/master/notes/nmt)

To get more specific. Below are the foundational seq2seq papers listed chronologically:
### Neural Machine Translation(NMT)

* Sequence to Sequence Learning with Neural Networks[[arXiv](https://arxiv.org/abs/1409.3215)] [[notes](https://github.com/iamsiva11/DLNLP-papernotes/blob/master/notes/nmt/seq2seq-with-Neural-Networks.md)] :clipboard:

* Neural Machine Translation by Jointly Learning to Align and Translate (2015)[[arXiv](https://arxiv.org/abs/1409.0473)] [[notes](https://github.com/iamsiva11/DLNLP-papernotes/blob/master/notes/nmt/nmt-by-Jointly-Learning-to-AlignandTranslate.md)] :clipboard:

* Effective Approaches to Attention-based Neural Machine Translation
[[arXiv](https://arxiv.org/abs/1508.04025)] [[notes](https://github.com/iamsiva11/DLNLP-papernotes/blob/master/notes/nmt/Effective-Approaches-to-Attention-based-nmt.md)] :clipboard:

* Massive Exploration of Neural Machine Translation Architectures[[arXiv](https://arxiv.org/pdf/1703.03906.pdf)] [[notes](https://github.com/iamsiva11/DLNLP-papernotes/blob/master/notes/nmt/Massive-exploration-NMT.md)] :clipboard:










