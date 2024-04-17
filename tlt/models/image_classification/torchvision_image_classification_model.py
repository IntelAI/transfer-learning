#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
from tqdm import tqdm

import torch

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.models.image_classification.pytorch_image_classification_model import PyTorchImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.utils.file_utils import read_json_file
from tlt.utils.platform_util import PlatformUtil

try:
    habana_import_error = None
    import habana_frameworks.torch.core as htcore
    is_hpu_available = True
except Exception as e:
    is_hpu_available = False
    habana_import_error = str(e)


class TorchvisionImageClassificationModel(PyTorchImageClassificationModel):
    """
    Class to represent a Torchvision pretrained model for image classification
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        torchvision_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/torchvision_image_classification_models.json"))
        if model_name not in torchvision_model_map.keys():
            raise ValueError("The specified Torchvision image classification model ({}) "
                             "is not supported.".format(model_name))

        PyTorchImageClassificationModel.__init__(self, model_name, **kwargs)

        self._classification_layer = torchvision_model_map[model_name]["classification_layer"]
        self._image_size = torchvision_model_map[model_name]["image_size"]
        self._original_dataset = torchvision_model_map[model_name]["original_dataset"]

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._distributed = False
        self._device = kwargs.get("device", "cpu")
        self._hub = "torchvision"
        self._enable_auto_mixed_precision = None

    def _model_downloader(self, model_name):
        downloader = ModelDownloader(model_name, hub=self._hub, model_dir=None, weights=self._original_dataset)
        model = downloader.download()
        return model

    def _get_hub_model(self, num_classes, ipex_optimize=True, extra_layers=None):
        if not self._model:
            self._model = self._model_downloader(self._model_name)

            if not self._do_fine_tuning:
                for param in self._model.parameters():
                    param.requires_grad = False

            # Do not apply a softmax activation to the final layer as with TF because loss can be affected
            if len(self._classification_layer) == 2:
                base_model = getattr(self._model, self._classification_layer[0])
                classifier = getattr(self._model, self._classification_layer[0])[self._classification_layer[1]]
                self._model.classifier = base_model[0: self._classification_layer[1]]
                num_features = classifier.in_features
                if extra_layers:
                    for layer in extra_layers:
                        self._model.classifier.append(torch.nn.Linear(num_features, layer))
                        self._model.classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                self._model.classifier.append(torch.nn.Linear(num_features, num_classes))
            else:
                classifier = getattr(self._model, self._classification_layer[0])
                if self._classification_layer[0] == "heads":
                    num_features = classifier.head.in_features
                else:
                    num_features = classifier.in_features

                if extra_layers:
                    setattr(self._model, self._classification_layer[0], torch.nn.Sequential())
                    classifier = getattr(self._model, self._classification_layer[0])
                    for layer in extra_layers:
                        classifier.append(torch.nn.Linear(num_features, layer))
                        classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                    classifier.append(torch.nn.Linear(num_features, num_classes))
                else:
                    classifier = torch.nn.Sequential(torch.nn.Linear(num_features, num_classes))
                    setattr(self._model, self._classification_layer[0], classifier)

            self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

            if ipex_optimize and not self._distributed:
                import intel_extension_for_pytorch as ipex
                ipex_dtype = torch.bfloat16 if self._enable_auto_mixed_precision else None
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer, dtype=ipex_dtype)

        self._num_classes = num_classes
        return self._model, self._optimizer

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, extra_layers=None, ipex_optimize=True,
              distributed=False, hostfile=None, nnodes=1, nproc_per_node=1, use_horovod=False, hvd_start_timeout=30,
              enable_auto_mixed_precision=None, device=None):
        """
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the model from torchvision and add on a fully-connected dense layer with linear activation
            based on the number of classes in the specified dataset. The model and optimizer are defined and trained
            for the specified number of epochs.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                epochs (int): Number of epochs to train the model (default: 1)
                initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
                early_stopping (bool): Enable early stopping if convergence is reached while training
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for evaluate. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make evaluation run faster and use less memory.
                    It is recommended to enable auto mixed precision when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the evaluation performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms.
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                seed (int): Optionally set a seed for reproducibility.
                extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a PyTorch model.
                    The input should be a list of integers representing the number and size of the layers,
                    for example [1024, 512] will insert two dense layers, the first with 1024 neurons and the
                    second with 512 neurons.
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to True.
                distributed (bool): Boolean flag to use distributed training. Defaults to False.
                hostfile (str): Name of the hostfile for distributed training. Defaults to None.
                nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
                nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
                    to 1.
                device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

            Returns:
                Trained PyTorch model object
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints,
                                 distributed, hostfile)

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
            # Gaudi is not compatible with IPEX
            print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
            ipex_optimize = False
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif self._device == "hpu" and is_hpu_available:
            if ipex_optimize:
                print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
                ipex_optimize = False

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        self._distributed = distributed

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("The extra_layers parameter must be a list of ints but found a list "
                                        "containing {}".format(type(layer)))

        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._set_seed(seed)

        # IPEX optimization can be suppressed with input ipex_optimize=False or
        # If are loading weights, the state dicts need to be loaded before calling ipex.optimize, so get the model
        # from torchvision, but hold off on the ipex optimize call.
        optimize = ipex_optimize and (False if initial_checkpoints else True)

        self._model, self._optimizer = self._get_hub_model(dataset_num_classes, ipex_optimize=optimize,
                                                           extra_layers=extra_layers)

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Call ipex.optimize now, since we didn't call it from _get_hub_model()
            if ipex_optimize and not distributed:
                import intel_extension_for_pytorch as ipex
                ipex_dtype = torch.bfloat16 if self._enable_auto_mixed_precision else None
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer, dtype=ipex_dtype)

        if distributed:
            try:
                saved_objects_dir = self.export_for_distributed(
                    export_dir=os.path.join(output_dir, 'tlt_saved_objects'),
                    train_data=dataset.train_subset,
                    val_data=dataset.validation_subset
                )
                batch_size = dataset._preprocessed['batch_size']
                self._fit_distributed(saved_objects_dir, hostfile, nnodes, nproc_per_node, epochs, batch_size,
                                      ipex_optimize, use_horovod, hvd_start_timeout)
            except Exception as err:
                print("Error: \'{}\' occured while distributed training".format(err))
            finally:
                self.cleanup_saved_objects_for_distributed()
        else:
            self._model.train()
            self._fit(output_dir, dataset, epochs, do_eval, early_stopping, lr_decay, enable_auto_mixed_precision)

        return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False, enable_auto_mixed_precision=None,
                 device=None):
        """
        Evaluate the accuracy of the model on a dataset.

        Args:
            enable_auto_mixed_precision (bool or None): Enable auto mixed precision for evaluate. Mixed precision
                uses both 16-bit and 32-bit floating point types to make evaluation run faster and use less memory.
                It is recommended to enable auto mixed precision when running on platforms that support
                bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                does not support bfloat16, it can be detrimental to the evaluation performance. If
                enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                running with Intel fourth generation Xeon processors, and disabled for other platforms.
            use_test_set (bool): If there is a validation subset, evaluation will be done on it (by default) or on
                the test set (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
                used for evaluation.
            device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")
        """
        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))
        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        if use_test_set:
            if dataset.test_subset:
                eval_loader = dataset.test_loader
                data_length = len(dataset.test_subset)
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_loader = dataset.validation_loader
            data_length = len(dataset.validation_subset)
        else:
            eval_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            print("The model has not been trained yet, so evaluation is being done using the original model ",
                  "and its classes")
            model = self._model_downloader(self._model_name)
            optimizer = self._optimizer_class(model.parameters(), lr=self._learning_rate)
            # We shouldn't need ipex.optimize() for evaluation
        else:
            model = self._model
            optimizer = self._optimizer

        # Do the evaluation
        device = torch.device(self._device)
        model = model.to(device)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(eval_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                if enable_auto_mixed_precision:
                    # Call model using the torch automatic mixed precision context when mixed precision is enabled
                    with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self._loss(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if self._device == "hpu" and is_hpu_available:
                htcore.mark_step()

        epoch_loss = running_loss / data_length
        epoch_acc = float(running_corrects) / data_length

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return [epoch_loss, epoch_acc]

    def predict(self, input_samples, return_type='class', enable_auto_mixed_precision=None, device=None):
        """
        Perform feed-forward inference and predict the classes of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            return_type (str): Using 'class' will return the highest scoring class (default), using 'scores' will
                               return the raw output/logits of the last layer of the network, using 'probabilities' will
                               return the output vector after applying a softmax function (so results sum to 1)
            device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

        Returns:
            List of classes, probability vectors, or raw score vectors

        Raises:
            ValueError: if the return_type is not one of 'class', 'probabilities', or 'scores'
        """
        return_types = ['class', 'probabilities', 'scores']
        if not isinstance(return_type, str) or return_type not in return_types:
            raise ValueError('Invalid return_type ({}). Expected one of {}.'.format(return_type, return_types))

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))
        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        if self._model is None:
            print("The model has not been trained yet, so predictions are being done using the original model")
            model = self._model_downloader(self._model_name)
            model = model.to(self._device)
            predictions = model(input_samples)
        else:
            self._model.eval()
            with torch.no_grad():
                if enable_auto_mixed_precision:
                    # Call model using the torch automatic mixed precision context when mixed precision is enabled
                    with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                        self._model = self._model.to(self._device)
                        predictions = self._model(input_samples)
                else:
                    self._model = self._model.to(self._device)
                    predictions = self._model(input_samples)
        if return_type == 'class':
            _, predicted_ids = torch.max(predictions, 1)
            return predicted_ids
        elif return_type == 'probabilities':
            # logic from torch.nn.functional _get_softmax_dim()
            dim = input_samples.dim()
            if dim == 0 or dim == 1 or dim == 3:
                dim = 0
            else:
                dim = 1
            return torch.nn.functional.softmax(predictions, dim=dim)
        else:
            return predictions
