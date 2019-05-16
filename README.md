 


<div>
    <br>
    <center><strong><h1>Higher Order Tutorial on Deep Learning!</h1></strong></center>
    <center><strong><h3>Yam Peleg</h3></strong></center>
<div>


------


<div>
    <center><img src="imgs/keras_logo_humans.png" width="30%"/>
    <h1>www.github.com/ypeleg/ExpertDL</h1></center>
<div>


------

<!-- #region -->
## 1. *How to tune in*?

If you wanted to listen to someone speaks three hours straight about deep learning, You could have done so by the comfort of your house. 

But you are here! Physically!

So...

This tutorial is extremely hands-on! You are strongly encouraged to play with it yourself!  

### Options: 


### $\varepsilon$.  Run the notebooks locally 
- `git clone https://github.com/ypeleg/ExpertDL`


- You might think that the goal of this tutorial is for you to play around with Deep Learning. This is wrong.

- **The Rreal Goal of the Tutorial is To give you the Flexibility to use this In your own domain!** 

Therefore, this is by far the best option if you can get it working! 

------


### a. Play with the _notebooks_ dynamically (on Google Colab) 

- Anyone can use the [colab.research.google.com/notebook](https://colab.research.google.com/notebook) website (by [clicking](XXX) on the icon bellow) to run the notebook in her/his web-browser. You can then play with it as long as you like! 
- For this tutorial:
[![Google Colab](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/ypeleg/ExpertDL)
------

### b. Play with the _notebooks_ dynamically (on MyBinder) 
[![Binder](https://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/github/ypeleg/ExpertDL)

Anyone can use the [mybinder.org](http://mybinder.org/) website (by [clicking](http://mybinder.org/v2/gh/github/ypeleg/ExpertDL) on the icon above) to run the notebook in her/his web-browser.
You can then play with it as long as you like, for instance by modifying the values or experimenting with the code.

### c. View the _notebooks_ statically. (if all else failed..)
- Either directly in GitHub: [ypeleg/ExpertDL](https://github.com/ypeleg/ExpertDL);
- Or on nbviewer: [notebooks](http://nbviewer.jupyter.org/github/ypeleg/ExpertDL/).
<!-- #endregion -->

---

<!-- #region -->
## What do I mean High Order?

In short [1], one can treat recent advancements of the field of deep learning as an increment of order (complexity-wise) where the components we use now in DL research were the whole experiments not long ago. 

Example: GANs involve training a neural networks on top of the output from another neural network. This can be viewed as a network of networks. 

Example: Some Reinforcement Learning algorithms (Mostly A3C) involves using a network for predicting the future reward of a state and using another network that based of that predicts the optimal action. Again a network of networks. 


In this tutorial we assume that we allready have deep learning networks ready for us as of the shelf tools and we use them to construct more complex algorithms.



[1]. Poking this with me opens the pandora box.. 
     We might cover this is in great detail at the end of the tutorial. Depends on time.
<!-- #endregion -->

# Outline at a glance

- **Part I**: **Introduction**

    - Intro Keras
        - Functional API    
       
    - Reinforcement Learning
        - Intro
        - Bandit 
        - Q learning
        - Policy Gradients

    - Generative Adversarial Networks
        - Intro
        - DCGAN
        - CGAN
        - WGAN
        
    - Embeddings
    
    - Advanced Natural Language Processing
        - Transformers
        - Elmo
        - Bert        

<!-- #region -->
## One More thing..


<img style ="width:70%;" src="images/matplotlib.jpg"/>


You are probably famillier with this.. so..

### The tachles.py file

In this tutorial many of the irrelevant details are hidden in a special file called "tachles.py".
Simply go:
<!-- #endregion -->

```python
import tachles
```

---


# Requirements


This tutorial requires the following packages:

- Python version 2.7.11 Or Python version 3.5
    - Other versions of Python should be fine as well. 
    - but.. *who knows*? :P
    
- `numpy` version 1.10 or later: http://www.numpy.org/
- `scipy` version 0.16 or later: http://www.scipy.org/
- `matplotlib` version 1.4 or later: http://matplotlib.org/
- `pandas` version 0.16 or later: http://pandas.pydata.org
- `scikit-learn` version 0.15 or later: http://scikit-learn.org
- `keras` version 2.0 or later: http://keras.io
- `tensorflow` version 1.0 or later: https://www.tensorflow.org
- `ipython`/`jupyter` version 4.0 or later, with notebook support

(Optional but recommended):

- `pyyaml`
- `hdf5` and `h5py` (required if you use model saving/loading functions in keras)
- **NVIDIA cuDNN** if you have NVIDIA GPUs on your machines.
    [https://developer.nvidia.com/rdp/cudnn-download]()

The easiest way to get (most) these is to use an all-in-one installer such as [Anaconda](http://www.continuum.io/downloads) from Continuum. These are available for multiple architectures.


---


### Python Version


I'm currently running this tutorial with **Python 3** on **Anaconda**

```python
!python --version
```

<!-- #region -->
### Configure Keras with tensorflow

1) Create the `keras.json` (if it does not exist):

```shell
touch $HOME/.keras/keras.json
```

2) Copy the following content into the file:

```
{
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "floatx": "float32",
    "image_data_format": "channels_last"
}
```
<!-- #endregion -->

```python
!cat ~/.keras/keras.json
```

---


# Test if everything is up&running


## 1. Check import

```python
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
```

```python
import keras
```

## 2. Check installeded Versions

```python
import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import IPython
print('iPython:', IPython.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)
```

```python
import keras
print('keras: ', keras.__version__)

# optional
import theano
print('Theano: ', theano.__version__)

import tensorflow as tf
print('Tensorflow: ', tf.__version__)
```

<br>
<h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>


---

