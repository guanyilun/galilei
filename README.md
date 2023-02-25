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
<a href="https://opensource.org/licenses/MPL-2.0">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
<a href="https://zenodo.org/badge/latestdoi/594445054"><img src="https://zenodo.org/badge/594445054.svg" alt="DOI"></a>

`galilei` is a python package that makes emulating a numerical functions easier and more composable. It supports multiple backends such as pytorch-based neural networks, GPy-based gaussian process regression, etc. As of now, it defaults to a jax+flax+optax backend which supports automatic differenciation of the emulated function and easy composibility with the rest of the jax-based eco-system.

The motivation of emulating a function is that sometimes computing a function could be a time consuming task, so one may need to find fast approximations of a function that's better than basic interpolation techniques. An emulated function, on the other hand, can runs blazingly fast on a normal GPU achieveing over many orders of magnitude speed up. This idea of emulating function is not new. In the field of cosmology we have powerful tools such as
[cosmopower](https://github.com/alessiospuriomancini/cosmopower) and its derived works such as [axionEmu](https://github.com/keirkwame/axionEmu), whose idea inspired this work. My aim in this work differs from the previous approach, as I intend to make a more generic and easily-composible functional emulator which can take any generic parametrized numerical function as an input and and return a function with the exact same signature that can be used as a drop-in replacement for the old function in existing code base without any modifications.

## Features
- Support multiple backends: `torch`, `sklearn`, `gpy` (for gaussian process regression), `jax`.
- Flexible: Able to emulate generic numerical functions.
- Automatic differenciable (supported by selected backends): emulated function is automatically differenciable and easily composible with jax-based tools.
- Easy to use: just add a decorator `@emulate` and use your emulated function as a drop-in replacement of your existing function in code-base without additional modification.
- Allow arbitrary transformation of function output before training through the use of `Preconditioner`.


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

You can also easily save your trained model with the `save` option
```python
@emulate(samples={
    'a': np.random.rand(100),
    'b': np.random.rand(100)
}, backend='sklearn', save="test.pkl")
def test(a=1, b=1):
    x = np.linspace(0, 10, 100)
    return np.sin(a*x) + np.sin(b*x)
```
and when you use it in production, simply load a pretrained model with
```python
@emulate(backend='sklearn', load="test.pkl")
def test(a=1, b=1):
    x = np.linspace(0, 10, 100)
    return np.sin(a*x) + np.sin(b*x)
```
and your function will be replaced with a fast emulated version.

For more detailed usage examples, see this notebook:
<a href="https://colab.research.google.com/drive/1_pvuAIqLUz4gV1vxytueb7AMR6Jmx-8n?usp=sharing">
<img src="https://user-content.gitlab-static.net/dfbb2c197c959c47da3e225b71504edb540e21d6/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="open in colab">
</a>
## Roadmap

* TODO add prebuild preconditioners
* TODO support downloading files from web
* TODO auto infer backend
* TODO chains of preconditioners

## Credits
This package was created with the [ppw](https://zillionare.github.io/python-project-wizard) tool. For more information, please visit the [project page](https://zillionare.github.io/python-project-wizard/).

If this package is helpful in your work, please consider citing:
```bibtex
@article{yguan_2023,
    title={galilei: a generic function emulator},
    DOI={10.5281/zenodo.7651315},
    publisher={Zenodo},
    author={Yilun Guan},
    year={2023},
    month={Feb}}
```

Free software: MIT
