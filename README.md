# Garrulus Dataset Library

Garrulus Dataset Libary (`gdl`) package provides tools to work with the Garrulus dataset. This particularly includes data pre-processing, benchmarking, task creation such as semantic segementation, object detection, and classification task.

The backend of `gdl` is based on PyTorch and [torchgeo](https://github.com/microsoft/torchgeo/tree/main).

## Installation

```
pip install git+https://git.inf.h-brs.de/garrulus/dfm/gdl@main
```

## Features and example usage

* Mask label generation from geopackages for `field-D`
* RasterDataset for semantic segmentation
* Area of interest (AOI) sampling (sample inside AOI or grid cells). This also allows to sample within predifined train, valid, test grid cells
* Semantic segmentation tasks

## Benchmark datasets

* `field-D` dataset benchmark


## WIP

* Support other fields
* Object detection task
* Classification task
* Multispectral data pre-processing
* More tutorials
* Example usage with PyTorch Lightning
* Large vision models
