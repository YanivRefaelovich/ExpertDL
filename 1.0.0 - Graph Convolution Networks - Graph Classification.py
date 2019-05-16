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
#     <center><strong><h3>1.0.0 - Graph Classification</h3></strong></center> 
# <div>

# ------

# ### Keras DGL - Node Classification:
# ##  `tl;dr:  MutliGraphCNN(output_dim, num_filters)([X,Adj])`
#
# Importing: 
# ```python
# from keras_dgl.layers import MutliGraphCNN
# ```
#
# Just like any keras model: 
# ```python
# output = MultiGraphCNN(100, num_filters, activation='elu')([X, Adj])
# output = MultiGraphCNN(100, num_filters, activation='elu')([output, Adj])
# output = Lambda(lambda x: K.mean(x, axis=1))(output)  
# ```

# ------

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
from tachles import fix_gcn_paths, load_mutag

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
fix_gcn_paths()
import keras_dgl
from keras_dgl.layers import MultiGraphCNN, MultiGraphAttentionCNN
from examples.utils import normalize_adj_numpy, evaluate_preds, preprocess_edge_adj_tensor

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## The MUTAG Dataset
#
# The MUTAG dataset is distributed baseline dataset for graph learning. It contains information about 340 complex molecules that are potentially carcinogenic, which is given by the isMutagenic property.
#
# The molecules can be classified as “mutagenic” or “not mutagenic”.

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
A, A_orig, X, Y, num_edge_features, num_graph_nodes, num_graphs, orig_num_graph_nodes, orig_num_graphs = load_mutag()
print X.shape, Y.shape, A.shape

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
import keras.backend as K
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation, Dropout, Input, Lambda
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
# plot_graph(A)
# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ----
#
# <span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/keras_dgl/layers/multi_graph_cnn_layer.py#L9)</span>
# ## MutliGraphCNN
#
# ```python
# MutliGraphCNN(output_dim, num_filters, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```
#
# MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same. For graph with arbitrary size, one can simply append appropriate zero rows or columns in adjacency matrix (and node feature matrix) based on max graph size in the dataset to achieve this uniformity.
#
# __Arguments__
#
# - __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
# - __num_filters__: Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
# - __activation__: Activation function to use
# .
# If you don't specify anything, no activation is applied
# (ie. "linear" activation: `a(x) = x`).
# - __use_bias__: Boolean, whether the layer uses a bias vector.
# - __kernel_initializer__: Initializer for the `kernel` weights matrix
# .
# - __bias_initializer__: Initializer for the bias vector
# .
# - __kernel_regularizer__: Regularizer function applied to
# the `kernel` weights matrix
# .
# - __bias_regularizer__: Regularizer function applied to the bias vector
# .
# - __activity_regularizer__: Regularizer function applied to
# the output of the layer (its "activation").
# .
# - __kernel_constraint__: Constraint function applied to the kernel matrix
# .
# - __bias_constraint__: Constraint function applied to the bias vector
# .
#
# __Input shapes__
#
# * __graph node feature matrix__ input as a 3D tensor with shape: `(batch_size, num_graph_nodes, input_dim)` corresponding to graph node input feature matrix for each graph.<br />
# * __graph_conv_filters__ input as a 3D tensor with shape: `(batch_size, num_filters*num_graph_nodes, num_graph_nodes)` <br />
# `num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian.<br />
#
# __Output shape__
#
# * 3D tensor with shape: `(batch_size, num_graph_nodes, output_dim)`	representing convoluted output graph node embedding matrix for each graph in batch size.<br />
#
#
#
# <span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/examples/multi_gcnn_graph_classification_example.py)</span>

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## The model itself

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
num_filters = num_edge_features
graph_conv_filters = preprocess_edge_adj_tensor(A, symmetric=True)

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
X_input = Input(shape=(X.shape[1], X.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

output = MultiGraphCNN(100, num_filters, activation='elu')([X_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphCNN(100, num_filters, activation='elu')([output, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  
output = Dense(Y.shape[1])(output)
output = Activation('softmax')(output)

nb_epochs = 200
batch_size = 169

model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
model.fit([X, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## Your Turn! 
# ### Run The same but this time with Attention CGNN!

# + {"cell_type": "markdown", "outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}, "colab_type": "code", "id": "3lOBizVa4rVt"}
# ## MultiGraphAttentionCNN
#
# ```python
# MutliGraphCNN(output_dim, num_filters, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```
#
# MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same. For graph with arbitrary size, one can simply append appropriate zero rows or columns in adjacency matrix (and node feature matrix) based on max graph size in the dataset to achieve this uniformity.
#
# __Arguments__
#
# - __output_dim__: Positive integer, dimensionality of each graph node feature output space (or also referred dimension of graph node embedding).
# - __num_filters__: Positive integer, number of graph filters used for constructing  __graph_conv_filters__ input.
# - __activation__: Activation function to use
# .
# If you don't specify anything, no activation is applied
#
# - __use_bias__: Boolean, whether the layer uses a bias vector.
# - __kernel_initializer__: Initializer for the `kernel` weights matrix
#
# - __bias_initializer__: Initializer for the bias vector
#
# - __kernel_regularizer__: Regularizer function applied to
# the `kernel` weights matrix
#
# - __bias_regularizer__: Regularizer function applied to the bias vector
#
# - __activity_regularizer__: Regularizer function applied to
# the output of the layer (its "activation").
#
# - __kernel_constraint__: Constraint function applied to the kernel matrix
#
# - __bias_constraint__: Constraint function applied to the bias vector
#
#
# __Input shapes__
#
# * __graph node feature matrix__ input as a 3D tensor with shape: `(batch_size, num_graph_nodes, input_dim)` corresponding to graph node input feature matrix for each graph.<br />
# * __graph_conv_filters__ input as a 3D tensor with shape: `(batch_size, num_filters*num_graph_nodes, num_graph_nodes)` <br />
# `num_filters` is different number of graph convolution filters to be applied on graph. For instance `num_filters` could be power of graph Laplacian.<br />
#
# __Output shape__
#
# * 3D tensor with shape: `(batch_size, num_graph_nodes, output_dim)`	representing convoluted output graph node embedding matrix for each graph in batch size.<br />
#
#
#
# <span style="float:right;">[[source]](https://github.com/vermaMachineLearning/keras-deep-graph-learning/blob/master/examples/multi_graph_attention_cnn_graph_classification_example.py)</span>

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
num_filters = 2
print A.shape

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
A_eye_tensor = []
for _ in range(orig_num_graphs):
    Identity_matrix = np.eye(orig_num_graph_nodes)
    A_eye_tensor.append(Identity_matrix)

A_eye_tensor = np.array(A_eye_tensor)
A_orig = np.add(A_orig, A_eye_tensor)
graph_conv_filters = preprocess_edge_adj_tensor(A_orig, symmetric=True)

# + {"outputId": "a3142dd2-4ff0-4bb6-a833-a7046f4e0596", "colab_type": "code", "id": "3lOBizVa4rVt", "colab": {"base_uri": "https://localhost:8080/", "height": 1295}}
# build model
X_input = Input(shape=(X.shape[1], X.shape[2]))
A_input = Input(shape=(A_orig.shape[1], A_orig.shape[2]))
graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]))

output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=2, attention_combine='concat', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([X_input, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = MultiGraphAttentionCNN(100, num_filters=num_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.5, activation='elu', kernel_regularizer=l2(5e-4))([output, A_input, graph_conv_filters_input])
output = Dropout(0.2)(output)
output = Lambda(lambda x: K.mean(x, axis=1))(output)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
output = Dense(Y.shape[1], activation='elu')(output)
output = Activation('softmax')(output)

nb_epochs = 500
batch_size = 169

model = Model(inputs=[X_input, A_input, graph_conv_filters_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit([X, A_orig, graph_conv_filters], Y, batch_size=batch_size, validation_split=0.1, epochs=nb_epochs, shuffle=True, verbose=1)
