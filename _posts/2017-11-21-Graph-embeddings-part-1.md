---
layout: post
title: Graphs and representation learning - Part I
published: true
---


![Knowledge graphs]({{site.baseurl}}/images/ge1-knowledge-graph.png)

I have been working in the area of Network Representation Learning(aka. graph embeddings) for nearly a year now. Specifically, my work on graph embeddings deals with Knowledge graphs. So, I want to paint a high level picture about graph embeddings in general with this blog post. This blog post is comprises of 2 sections - overview of representation learning, overview of graph embeddings. Which helps to build a foundation and big picture of the in-depth.

Moreover, I'm planning to write this as a 2-part series. That being said, subsequent part of the series will cover graph embeddings in depth. With the first part being this one, which you are currently reading right now. Second part will go in detail about the current research, state of art graph embedding techniques(approaches like random walk based , deep learning based, etc) in detail. The other subsequent parts I haven't planned now would focus on the applied, code implementations side of graph embeddings, detailed write-up on the individual graph embeddings technique.

Pedro Domingos, a CS professor at the University of Washington published a brief and immensely readable paper in 2012[1] that helpfully decomposed machine learning into three components: Representation, Evaluation, and Optimization. In this particular blog post, we are very much concerned about the representation part.

![Machine learning three components ]({{site.baseurl}}/images/repr-opt-eval.png)

Many information processing tasks can be very easy or very diﬃcult depending on how the information is represented. This is a general principle applicable to daily life, to computer science in general, and to machine learning. For example, it is straightforward for a person to divide 210 by 6 using long division. The task becomes considerably less straightforward if it is instead posed using the Roman numeral representation of the numbers. Most modern people asked to divide CC by VI would begin by converting the numbers to the Arabic numeral representation, permitting long division procedures that make use of the place value system.[3]

# Representation Learning

Representation is basically representing the input data(be image, speech, or video for that matter) in a way the learning algorithm can easily process. Representation Learning is using learning algorithms to derive good features or representation automatically, instead of traditional hand-engineered features. Specifically, during learning it transforms the raw data input to a representation that can be effectively exploited by learning algorithms. Hence this unsupervised feature learning obviates the need for manual feature engineering, which allows a machine to both learn the features and use them to perform a specific task.

While looking back at older machine learning algorithms, they made great progress in training classification, regression and recognition systems when "good" representations, or features, of input data are available. They rely on the input being a feature and learn a classifier, regressor, etc on top of that. Most of these features were hand crafted, meaning, they were designed by humans. However, much human effort was spent on designing good features which are usually knowledge-based and engineered by domain experts over years of trial and error. A natural question to ask then is "Can we automate the learning of useful features from raw data?". And the answer lied on representation learning(aka. feature engineering/learning).

And coming to deep  learning, it exploits this concept by its very nature. Multilayer neural networks can be used to perform feature learning, since they learn a representation of increasing complexity/abstraction of their input at the hidden layer(s) which is subsequently used for classification or regression at the output layer. While the representations or features learned corresponds to the hidden stochastic neurons. Specifically, Restricted Boltzmann machines, auto encoders, deep belief networks, convolutional neural networks are well known architectures for representation Learning. These learnt features are increasingly more informative through layers towards the machine learning task that we intend to perform (e.g. classification).

![Focus of this blog post - input to algorithm]({{site.baseurl}}/images/input-algo-2.png)

So basically, we have two learning tasks now. First, we let the deep network discover the features and next place our preferred learning model to perform your task. Simple!

Yoshua Bengio is one of the leaders in deep learning; although he began with a strong interest in the automatic feature learning that large neural networks are capable of achieving. 

![yoshua Bengio and representation learning ]({{site.baseurl}}/images/Yoshua-Bengio.jpg)

In his 2012 paper titled “Deep Learning of Representations for Unsupervised and Transfer Learning”[2] he commented:

> "Deep learning algorithms seek to exploit the unknown structure in the input distribution in order to discover good representations, often at multiple levels, with higher-level learned features defined in terms of lower-level features"

Furthermore, Deep models follow this modern view of representation learning[4] - learn representations (a.k.a. causes or features) that are

* Hierarchical: representations at increasing levels of abstraction
* Distributed: information is encoded by a multiplicity of causes
* Shared among tasks
* Sparse: enforcing neural selectivity
* Characterized by simple dependencies: a simple (linear) combination of the representations should be sufficient to generate data


![Success of deep learning ]({{site.baseurl}}/images/ge1-success-of-deep-learning.png)

More often than not, how rich your input representation is has huge bearing on the quality of your downstream learning algorithms. Now let's take a specific domain of natural language processing(NLP) and look at how representation learning and deep learning exploits this task. 

If we look at the traditional legacy techniques of representing text(before deep learning) were one-hot encoding, Bag of words, N-gram, TF-IDF. Since these archaic techniques treat words as atomic symbols, every 2 words are equally apart. They don’t have any notion of either syntactic or semantic similarity between parts of language. This is one of the chief reasons for poor/mediocre performance of NLP based models. But this has changed dramatically in past few years.

![one hot , bag of words]({{site.baseurl}}/images/traditional-nlp-representation-2.png)

Representation of words as vectors(aka. embedding) is a very good example of representation learning. You could read about Word2Vec or GloVE where words are represented as vectors in a hyperspace. Due to its popularity, word2vec has become the de facto "Hello, world!" application of representation learning. When applying deep learning to natural language processing (NLP) tasks, the model must simultaneously learn several language concepts: the meanings of words, how words are combined to form concepts (i.e., syntax), how concepts relate to the task at hand. Word embeddings helps you achieve this -> king - man + woman = queen.

![Word vector king queen]({{site.baseurl}}/images/king-queen-word-vector.png)

That is, when you perform this calculation over the vectors of king, man and woman, you would have a vector which would be very close to that of queen. This is what deep learning for word representation has helped to achieve.

![Embedding values for many words]({{site.baseurl}}/images/word-vectors-many-word-num.png)

Mathematically, embedding maps information entities into a dense, real-valued, and low-dimensional vectors. Specifically in NLP, it maps between space with one dimension per linguistic unit (character, morpheme, word, phrase, paragraph, sentence, document) to a continuous vector space with much lower dimension. Thus the “meaning” of linguistic unit is represented by a vector of real numbers.

![Embedding example for a single word - man]({{site.baseurl}}/images/man-number-vector.png)

And in the field of computer vision; deep learning and representation learning provided state of art results for vision tasks like image classification, object detection,etc. Classical examples of features in computer vision include Harris, SIFT, LBP, etc. Images can be represented using these features and learning algorithms can be applied on top of that.  We can have a neural network which takes the image as an input and outputs a vector, which is the feature representation of the image. This is the representation learner. This be followed by another neural network that acts as the classifier, regressor,etc.

![traditional image representations ]({{site.baseurl}}/images/learning-features-shallow-way-deep-way-image.png)

A wheel has a geometric shape, but its image may be complicated by shadows falling on the wheel, the sun glaring off the metal parts of the wheel, the fender of the car or an object in the foreground obscuring part of the wheel, and so on. One solution to this problem is to use ML to discover not only the mapping from representation to output but also the representation to itself. This approach is called representation learning. Instead of manually describing the wheel; like, say it should be circular, be black in colour, have treads, etc. But these are all hand crafted features and may not generalize to all situations. For example, if you look at the wheel from a different angle, it might be oval in shape. Or the lighting may cause it to have lighter and darker patches. These kinds of variations are hard to account for manually. Instead, we can let the representation learning neural network learn them from data by giving it several positive and negative examples of a wheel and training it end to end.

That's a wrap on the representation learning part. Let's get deeper into Graph embeddings now (our topic of interest in this blog series).

![Eucledian data visualisation]({{site.baseurl}}/images/ge1-Eucledian-data-grids-sequences.png)

Specifically, in a multitude of different fields, such as: Biology, Physics, Network Science, Recommender Systems and Computer Graphics; one may have to deal with data defined on non-Euclidean domains (i.e. graphs and manifolds). The adoption of Deep Learning in these particular fields has been lagging behind until very recently, primarily since the non-Euclidean nature of data makes the definition of basic operations (such as convolution) rather elusive. Geometric Deep Learning deals in this sense with the extension of Deep Learning techniques to graph/manifold structured data. Geometric Deep Learning is one of the most emerging fields of the Machine Learning community.

![Graphs pictorial representation]({{site.baseurl}}/images/data-looks-like-graph.png)


In the last decade, deep learning techniques for representation learning disrupted the field of Computer Vision, speech and text, which we have seen above. That said, research on DL techniques has mainly focused so far on data defined on Euclidean domains (i.e. grids, sequences). There have been very few studies involving representation learning on network data (for example, social network data). Recently, methods which use the representation of graph nodes in vector space have gained traction from the research community.

### So, how can we represent non-euclidean data like graphs?

