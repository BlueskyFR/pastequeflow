<p align="center"><a href="https://git.io/pastequeflow" target="_blank"><img width="350" src=".images/PastequeFlow-logo.png" alt="PastequeFlow logo" /></a></p>

:watermelon: This repository hosts PastequeFlow's source code!

PastequeFlow is a wrapper around TensorFlow and Keras, using an Object-Oriented approach.

Keras is used first and TensorFlow is used directly when more appropriate.

> PastequeFlow is undergoing heavy development and is unstable right now. The framework is not yet ready to use anywhere.

As of now, it is recommended to develop it using git submodules.

## Dependencies

The dependencies can be loaded into a (existing or not) conda environment by importing `environment.yml`:

```bash
# This will create/add the dependencies to the "tf" env (default name);
# you can customize the name of the environment with `-n <name>`
# and the path it will be install to using `-p <path>`
conda env update --file environment.yml
```
