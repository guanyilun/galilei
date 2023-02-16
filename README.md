# galilei

<a href="https://pypi.python.org/pypi/galilei">
    <img src="https://img.shields.io/pypi/v/galilei.svg"
        alt = "Release Status">
</a>
<a href="https://github.com/guanyilun/galilei/actions">
    <img src="https://github.com/guanyilun/galilei/actions/workflows/release.yml/badge.svg?branch=master" alt="CI Status">
</a>
<a href="https://github.com/guanyilun/galilei/actions">
    <img src="https://github.com/guanyilun/galilei/actions/workflows/dev.yml/badge.svg?branch=master" alt="CI Status">
</a>
<a href="https://guanyilun.github.io/galilei/">
    <img src="https://img.shields.io/website/https/guanyilun.github.io/galilei/index.html.svg?label=docs&down_message=unavailable&up_message=available" alt="Documentation Status">
</a>
<a href="https://codecov.io/gh/guanyilun/galilei" >
 <img src="https://codecov.io/gh/guanyilun/galilei/branch/master/graph/badge.svg?token=0C3DT4IWP5"/>
</a>
<a href="https://opensource.org/licenses/MPL-2.0">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>


`galilei` is a software package that makes emulating a function easier. The motivation of emulating a function is that sometimes computing a function could be a time consuming task, so one may need to find fast approximations of a function that's better than basic interpolation techniques. It builds on the ideas of
[cosmopower](https://github.com/alessiospuriomancini/cosmopower) and [axionEmu](https://github.com/keirkwame/axionEmu), with an aim to be as generic and flexible as possible on the emulating target. As such, `galilei` can take any generic parametrized function that returns an array without a need to know its implementation detail.

## Installation
```
pip install galilei
```

## Basic usage
Suppose that we have an expensive function that we want to emulate
```python
def test(a=1, b=1):
    x = np.linspace(0, 10, 100)
    return np.sin(a*x) + np.sin(b*x)
```
If you want to emulate this function, you can simply add a decorator `@emulate` and supply the parameters that you want to evaluate this function at to build up the training data set.

```python
from galilei import emulate

@emulate(samples={
    'a': np.random.rand(1000),
    'b': np.random.rand(1000)
})
def test(a=1, b=1):
    x = np.linspace(0, 10, 100)
    return np.sin(a*x) + np.sin(b*x)
```
Here we are just making 1000 pairs of random numbers from 0 to 1 to train our function. When executing these lines, the emulator will start training, and once it is done, the original `test` function will be automatically replaced with the emulated version and should behave in the same way, except much faster!
```
Training emulator...
100%|██████████| 500/500 [00:09<00:00, 50.50it/s, loss=0.023]
Ave Test loss: 0.025
```
![Comparison](https://github.com/guanyilun/galilei/raw/master/data/demo.png)

For more detailed usage examples, see this notebook:
<a href="https://colab.research.google.com/drive/1_pvuAIqLUz4gV1vxytueb7AMR6Jmx-8n?usp=sharing">
<img src="https://user-content.gitlab-static.net/dfbb2c197c959c47da3e225b71504edb540e21d6/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="open in colab">
</a>
## Features

* TODO support saving trained model and load pretrained model
* TODO add Gpy backend for gaussian process regression

## Credits
This package was created with the [ppw](https://zillionare.github.io/python-project-wizard) tool. For more information, please visit the [project page](https://zillionare.github.io/python-project-wizard/).

Free software: MIT
