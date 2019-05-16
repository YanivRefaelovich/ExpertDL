# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# ------

# <div> 
#     <center><h5>Higher Order Tutorial on Deep Learning</h5></center>
#     <center><strong><h2>Graph Convolution Networks</h2></strong></center>
#     <center><strong><h3>1.0.1 - Node Classification</h3></strong></center> 
# <div>

# ------

# ### Keras DGL - Node Classification:
# ##  `tl;dr:  GraphCNN(output_dim, num_filters, graph_conv_filters)`
#
# Importing: 
# ```python
# from keras_dgl.layers import GraphCNN
# ```
#
# Just like any keras model: 
# ```python
# model = Sequential()
# model.add(GraphCNN(16, 2, graph_conv_filters, input_shape=(X.shape[1],)))
# model.add(GraphCNN(Y.shape[1], 2, graph_conv_filters))
# model.add(Activation('softmax'))
# ```

# ------

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# # Graph Node Classification

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# ### Motivation :
#
# There is a lot of data out there that can be represented in the form of a graph
# in real-world applications like in Citation Networks, Social Networks (Followers
# graph, Friends network, … ), Biological Networks or Telecommunications. <br>
# Using Graph extracted features can boost the performance of predictive models by
# relying of information flow between close nodes. However, representing graph
# data is not straightforward especially if we don’t intend to implement
# hand-crafted features.<br> In this post we will explore some ways to deal with
# generic graphs to do node classification based on graph representations learned
# directly from data.
#
# ### Dataset :
#
# The [Cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) citation network
# data set will serve as the base to the implementations and experiments
# throughout this post. Each node represents a scientific paper and edges between
# nodes represent a citation relation between the two papers.<br> Each node is
# represented by a set of binary features ( Bag of words ) as well as by a set of
# edges that link it to other nodes.<br> The dataset has **2708** nodes classified
# into one of seven classes. The network has **5429** links. Each Node is also
# represented  by a binary word features indicating the presence of the
# corresponding word. Overall there is **1433** binary (Sparse) features for each
# node. In what follows we *only* use **140** samples for training and the rest
# for validation/test.

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# ### Problem Setting :
#
# ![](https://cdn-images-1.medium.com/max/1600/1*klF4yon9ZpP6oZ0kvO86QA.png)
#
# **Problem** : Assigning a class label to nodes in a graph while having few
# training samples.<br> **Intuition**/**Hypothesis** : Nodes that are close in the
# graph are more likely to have similar labels.<br> **Solution** : Find a way to
# extract features from the graph to help classify new nodes.
#
# ### Proposed Approach :
#
# <br>
#
# **Baseline Model :**
#
# ![](https://cdn-images-1.medium.com/max/1600/1*nlDeQPW2ABhtwjoSI2dvWQ.png)
#
# We first experiment with the simplest model that learn to predict node classes
# using only the binary features and discarding all graph information.<br> This
# model is a fully-connected Neural Network that takes as input the binary
# features and outputs the class probabilities for each node.
#
# #### **Baseline model Accuracy : 53.28%**
# Source: https://github.com/CVxTz/graph_classification

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# **Adding Graph features :**
#
# One way to automatically learn graph features by embedding each node into a
# vector by training a network on the auxiliary task of predicting the inverse of
# the shortest path length between two input nodes like detailed on the figure and
# code snippet below :
#
# ![](https://cdn-images-1.medium.com/max/1600/1*PP_y_YhkKFYpzkj7szhnaw.png)

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# The next step is to use the pre-trained node embedding as input to the
# classification model. We also add the an additional input which is the average
# binary features of the neighboring nodes using distance of learned embedding
# vectors.
#
# The resulting classification network is described in the following figure :
#
# ![](https://cdn-images-1.medium.com/max/1600/1*xc99u2ejelSXNPKPmh-Nrw.png)
#
# <span class="figcaption_hack">Using pretrained embeddings to do node classification</span>
#
#

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# **Improving Graph feature learning :**
#
# We can look to further improve the previous model by pushing the pre-training
# further and using the binary features in the node embedding network and reusing
# the pre-trained weights from the binary features  in addition to the node
# embedding vector. This results in a model that relies on more useful
# representations of the binary features learned from the graph structure.
#
# ![](https://cdn-images-1.medium.com/max/1600/1*bEy9ua6jTBdkFGrrfvxpiA.png)
#

# + {"cell_type": "markdown", "slideshow": {"slide_type": "slide"}}
# # Graph Neural Networks

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
#
# Mathematically, the GCN model follows this formula:
#
# $H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$
#
# Here, $H^{(l)}$ denotes the $l^{th}$ layer in the network,
# $\sigma$ is the non-linearity, and $W$ is the weight matrix for
# this layer. $D$ and $A$, as commonly seen, represent degree
# matrix and adjacency matrix, respectively. The ~ is a renormalization trick
# in which we add a self-connection to each node of the graph, and build the
# corresponding degree and adjacency matrix.  The shape of the input
# $H^{(0)}$ is $N \times D$, where $N$ is the number of nodes
# and $D$ is the number of input features. We can chain up multiple
# layers as such to produce a node-level representation output with shape
# $N \times F$, where $F$ is the dimension of the output node
# feature vector.
#
# The equation can be efficiently implemented using sparse matrix
# multiplication kernels (such as Kipf's
# `https://github.com/tkipf/pygcn`). The above DGL implementation
# in fact has already used this trick due to the use of builtin functions. To
# understand what is under the hood, please read the tutorial on page rank specified in this repository.
#
# __References__: <br />
# [1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016). <br />
# [2] Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." In Advances in Neural Information Processing Systems, pp. 3844-3852. 2016. <br />
# [3] Simonovsky, Martin, and Nikos Komodakis. "Dynamic edge-conditioned filters in convolutional neural networks on graphs." In Proc. CVPR. 2017. <br />

# + {"language": "bash", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# if [ ! -d "keras-deep-graph-learning" ] ; then git clone https://github.com/ypeleg/keras-deep-graph-learning; fi

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
from tachles import fix_gcn_paths, load_cora

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
fix_gcn_paths()

import keras_dgl
from keras_dgl.layers import GraphCNN, GraphAttentionCNN

from examples.utils import normalize_adj_numpy, evaluate_preds

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## The CORA Dataset
# The dataset used in this demo can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
#
# The following is the description of the dataset:
# > The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
# > The citation network consists of 5429 links. Each publication in the dataset is described by a
# > 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
# > The dictionary consists of 1433 unique words. The README file in the dataset provides more details.
#
# Download and unzip the cora.tgz file to a location on your computer and set the `data_dir` variable to
# point to the location of the dataset (the directory containing "cora.cites" and "cora.content").

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
X, Y_train, Y_test, A, train_idx, val_idx, test_idx, train_mask = load_cora()
print X.shape, Y_train.shape, Y_test.shape

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
import keras.backend as K
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras.regularizers import l2
from keras.optimizers import Adam


# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
def plot_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    nx.draw_networkx(gr, ax=ax, with_labels=False, node_size=5, width=.5)
    ax.set_axis_off()
    plt.show()
    plt.close()
# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
print X[0]
plot_graph(A)
# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## GraphCNN
#
# ```python
# GraphCNN(output_dim, num_filters, graph_conv_filters,  activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```
#
# GraphCNN layer assumes a fixed input graph structure which is passed as a layer argument. As a result, the input order of graph nodes are fixed for the model and should match the nodes order in inputs. Also, graph structure can not be changed once the model is compiled. This choice enable us to use Keras Sequential API but comes with some constraints (for instance shuffling is not  possible anymore in-or-after each epoch).<br />
#
#
# __Arguments__
#
# - __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
# - __num_filters__: Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
# - __graph_conv_filters__ input as a 2D tensor with shape: `(num_filters*num_graph_nodes, num_graph_nodes)`<br />
# `num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian. Here list of graph convolutional matrices are stacked along second-last axis.<br />
# - __activation__: Activation function to use
# (see [activations](https://keras.io/activations)).
# If you don't specify anything, no activation is applied
# (ie. "linear" activation: `a(x) = x`).
# - __use_bias__: Boolean, whether the layer uses a bias vector.
# - __kernel_initializer__: Initializer for the `kernel` weights matrix
# (see [initializers](https://keras.io/initializers)).
# - __bias_initializer__: Initializer for the bias vector
# (see [initializers](https://keras.io/initializers)).
# - __kernel_regularizer__: Regularizer function applied to
# the `kernel` weights matrix
# (see [regularizer](https://keras.io/regularizers)).
# - __bias_regularizer__: Regularizer function applied to the bias vector
# (see [regularizer](https://keras.io/regularizers)).
# - __activity_regularizer__: Regularizer function applied to
# the output of the layer (its "activation").
# (see [regularizer](https://keras.io/regularizers)).
# - __kernel_constraint__: Constraint function applied to the kernel matrix
# (see [constraints](https://keras.io/constraints/)).
# - __bias_constraint__: Constraint function applied to the bias vector
# (see [constraints](https://keras.io/constraints/)).
#
#
#
# __Input shapes__
#
# * 2D tensor with shape: `(num_graph_nodes, input_dim)` representing graph node input feature matrix.<br />
#
#
# __Output shape__
#
# * 2D tensor with shape: `(num_graph_nodes, output_dim)`	representing convoluted output graph node embedding (or signal) matrix.<br />
#
#
#
# ----
# ## Remarks
#
# __Why pass graph_conv_filters as a layer argument and not as an input in GraphCNN?__<br />
# The problem lies with keras multi-input functional API. It requires --- all input arrays (x) should have the same number of samples i.e.,  all inputs first dimension axis should be same. In special cases the first dimension of inputs could be same, for example check out Kipf .et.al.  keras implementation [[source]](https://github.com/tkipf/keras-gcn/blob/master/kegra/train.py). But in cases such as a graph recurrent neural networks this does not hold true.
#
#   
# __Why pass graph_conv_filters as 2D tensor of this specific format?__<br />
# Passing  graph_conv_filters input as a 2D tensor with shape: `(K*num_graph_nodes, num_graph_nodes)` cut down few number of tensor computation operations.
#
# __References__: <br />
# [1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016). <br />
# [2] Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." In Advances in Neural Information Processing Systems, pp. 3844-3852. 2016. <br />
# [3] Simonovsky, Martin, and Nikos Komodakis. "Dynamic edge-conditioned filters in convolutional neural networks on graphs." In Proc. CVPR. 2017. <br />
#
#
# <span style="float:right;">[[source]](https://github.com/ypeleg/keras-deep-graph-learning/blob/master/examples/gcnn_node_classification_example.py)</span>

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
class EdgeEval(Callback):
    def __init__(self):
        super(EdgeEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        Y_pred = model.predict(X, batch_size=A.shape[0])
        _, train_acc = evaluate_preds(Y_pred, [Y_train], [train_idx])
        _, test_acc = evaluate_preds(Y_pred, [Y_test], [test_idx])
        print("Epoch: {:04d}".format(epoch), "train_acc= {:.4f}".format(train_acc[0]), "test_acc= {:.4f}".format(test_acc[0]))    
# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## The model itself

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
A_norm = normalize_adj_numpy(A, True)
num_filters = 2
graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
print graph_conv_filters.shape

graph_conv_filters = K.constant(graph_conv_filters)

# Build Model
inp = Input(shape=(X.shape[1],))
x = GraphCNN(16, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(inp)
x = Dropout(0.2)(x)
x = GraphCNN(Y_train.shape[1], num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4))(x)
x = Activation('softmax')(x)

model = Model(inputs = inp, outputs = x)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
model.summary()

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=500, shuffle=False, callbacks=[EdgeEval()], verbose=1)

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# <span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/graph_attention_cnn_layer.py#L10)</span>
# ## GraphAttentionCNN
#
# ```python
# GraphAttentionCNN(output_dim, adjacency_matrix, num_filters=None, graph_conv_filters=None, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```
#
# GraphAttention layer assumes a fixed input graph structure which is passed as a layer argument. As a result, the input order of graph nodes are fixed for the model and should match the nodes order in inputs. Also, graph structure can not be changed once the model is compiled. This choice enable us to use Keras Sequential API but comes with some constraints (for instance shuffling is not  possible anymore in-or-after each epoch). See further [remarks below](http://127.0.0.1:8000/Layers/Convolution/graph_conv_layer/#remarks) about this specific choice.<br />
#
#
# __Arguments__
#
# - __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
# - __adjacency_matrix__: input as a 2D tensor with shape: `(num_graph_nodes, num_graph_nodes)` with __diagonal values__ equal to 1.<br />
# - __num_filters__: None or Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
# - __graph_conv_filters__: None or input as a 2D tensor with shape: `(num_filters*num_graph_nodes, num_graph_nodes)`<br />
# `num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian. Here list of graph convolutional matrices are stacked along second-last axis.<br />
# - __activation__: Activation function to use
# (see [activations](../activations.md)).
# If you don't specify anything, no activation is applied
# (ie. "linear" activation: `a(x) = x`).
# - __use_bias__: Boolean, whether the layer uses a bias vector (recommended setting is False for this layer).
# - __kernel_initializer__: Initializer for the `kernel` weights matrix
# (see [initializers](../initializers.md)).
# - __bias_initializer__: Initializer for the bias vector
# (see [initializers](../initializers.md)).
# - __kernel_regularizer__: Regularizer function applied to
# the `kernel` weights matrix
# (see [regularizer](../regularizers.md)).
# - __bias_regularizer__: Regularizer function applied to the bias vector
# (see [regularizer](../regularizers.md)).
# - __activity_regularizer__: Regularizer function applied to
# the output of the layer (its "activation").
# (see [regularizer](../regularizers.md)).
# - __kernel_constraint__: Constraint function applied to the kernel matrix
# (see [constraints](https://keras.io/constraints/)).
# - __bias_constraint__: Constraint function applied to the bias vector
# (see [constraints](https://keras.io/constraints/)).
#
#
#
# __Input shapes__
#
# * 2D tensor with shape: `(num_graph_nodes, input_dim)` representing graph node input feature matrix.<br />
#
#
# __Output shape__
#
# * 2D tensor with shape: `(num_graph_nodes, output_dim)`	representing convoluted output graph node embedding (or signal) matrix.<br />
#
#
# <span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/examples/graph_attention_cnn_node_classification_example.py)</span>

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
A_norm = normalize_adj_numpy(A, True)
num_filters = 2
graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
print graph_conv_filters.shape

graph_conv_filters = K.constant(graph_conv_filters)


# Build Model
inp = Input(shape=(X.shape[1],))
x = GraphAttentionCNN(8, A, num_filters, graph_conv_filters, num_attention_heads=8, attention_combine='concat', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4))(inp)
x = Dropout(0.6)(x)
x = GraphAttentionCNN(Y_train.shape[1], A, num_filters, graph_conv_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4))(x)
x = Activation('softmax')(x)

model = Model(inputs = inp, outputs = x)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
model.summary()

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
model.fit(X, Y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=500, shuffle=False, callbacks=[EdgeEval()], verbose=1)
