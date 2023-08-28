*Note: You may find it easier to read about Intel Transfer Learning tool, follow the Get
Started guide, and browse the API material from our published documentation site
https://intelai.github.io/transfer-learning.*

<!-- SkipBadges -->

# Intel® Transfer Learning Tool

Transfer learning workflows use the knowledge learned by a pre-trained model on
a large dataset to improve the performance of a related problem with a smaller
dataset.

## What is Intel® Transfer Learning Tool

Intel® Transfer Learning Tool makes it easier and faster for you to
create transfer learning workflows across a variety of AI use cases. Its
open-source Python\* library leverages public pretrained model hubs,
Intel-optimized deep learning frameworks, and your custom dataset to efficiently
generate new models optimized for Intel hardware.

This project documentation provides information, resource links, and instructions for the Intel
Transfer Learning Tool as well as Jupyter\* notebooks and examples that
demonstrate its usage.

**Features:**
* Supports PyTorch\* and TensorFlow\*
* Select from over [100 image classification and text classification models](Models.md) from
  Torchvision, PyTorch Hub, TensorFlow Hub, Keras, and Hugging Face
* Use your own custom dataset or get started quickly with [built-in datasets](DATASETS.md)
* Automatically create a trainable classification layer customized for your dataset
* Pre-process your dataset using scaling, cropping, batching, and splitting
* Use APIs for prediction, evaluation, and benchmarking
* Export your model for deployment or resume training from checkpoints

**Intel Optimizations:**
* Boost performance with Intel® Optimization for TensorFlow and Intel® Extension for PyTorch
* Quantize to INT8 to reduce model size and speed up inference using Intel® Neural Compressor
* Optimize model for FP32 inference using Intel Neural Compressor
* Reduce training time with auto-mixed precision for select hardware platforms
* Further reduce training time with multinode training for PyTorch

## How the Intel Transfer Learning Tool Works

The Intel Transfer Learning Tool lets you train AI models with TensorFlow or
PyTorch using either no-code command line interface (CLI) commands at a bash
prompt, or low-code application programming interface (API) calls from a Python
script.

Use your own dataset or select an existing image or text classification dataset listed in the
[public datasets](DATASETS.md) documentation. Construct your own CLI or API commands for training, evaluation,
and optimization using the TensorFlow or PyTorch framework, and finally export
your saved model optimized for inference on Intel CPUs.

An overview of the Intel Transfer Learning Tool flow is shown in this
figure:

<p align="center"><b>Intel Transfer Learning Tool Flow</b></p>

<img alt="Intel Transfer Learning Tool Flow" title="Intel Transfer Learing Tool Flow" src="images/TLT-tool_flow.svg" width="600">

## Get Started

Check out the [Get Started Guide](GetStarted.md) which will walk you through the
steps to check system requirements, install, and then run the tool with a couple of
examples showing no-code CLI and low-code API approaches. After that, you can check out
these additional CLI and API [Examples](examples/README.md).

<!-- ExpandGetStarted-Start -->
As described in the [Get Started Guide](GetStarted.md), once you have a Python 
environment set up, you do a basic install of the Intel Transfer Learning
Tool. Here are some examples of commands you will find in the [Get Started Guide](GetStarted.md):

```
pip install intel-transfer-learning-tool
```

Then you can use the Transfer Learning Tool CLI interface (tlt) to train a
TensorFlow image classification model (resnet_v1_50), download and use an
existing built-in dataset (tf_flowers), and save the trained model to
`/tmp/output` using this one command:

```
tlt train --framework tensorflow --model-name resnet_v1_50 --dataset-name tf_flowers \
   --output-dir /tmp/output --dataset-dir /tmp/data
```

Use `tlt --help` to see the list of CLI commands.  More detailed help for each
command can be found using, for example, `tlt train --help`.

<!-- ExpandGetStarted-End -->

## Support

The Intel Transfer Learning Tool team tracks bugs and enhancement requests using
[GitHub issues](https://github.com/IntelAI/transfer-learning-tool/issues). Before submitting a
suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

See [Legal Information](Legal.md) for Disclaimers, Trademark, and Licensing information.
