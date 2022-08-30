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
# SPDX-License-Identifier: EPL-2.0
#

import os
from torchvision import datasets

from tlt.datasets.pytorch_dataset import PyTorchDataset
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset


class PyTorchCustomImageClassificationDataset(ImageClassificationDataset, PyTorchDataset):
    """
    A custom image classification dataset that can be used with PyTorch models. Note that the
    directory of images is expected to be organized with subfolders for each image class. Each subfolder should
    contain .jpg images for the class. The name of the subfolder will be used as the class label.

    .. code-block:: text

        dataset_dir
          ├── class_a
          ├── class_b
          └── class_c
        
    Args:
        dataset_dir (str): Directory where the data is located. It should contain subdirectories with images for
                           each class.
        dataset_name (str): optional; Name of the dataset. If no dataset name is given, the dataset_dir folder name
                            will be used as the dataset name.
        num_workers (int): optional; Number of processes to use for data loading, default is 0

    Raises:
        FileNotFoundError if dataset directory does not exist

    """

    def __init__(self, dataset_dir, dataset_name=None, num_workers=0, shuffle_files=True):
        """
        Class constructor
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError("The dataset directory ({}) does not exist".format(dataset_dir))

        # The dataset name is only used for informational purposes. If one isn't given, use the directory name
        if not dataset_name:
            dataset_name = os.path.basename(dataset_dir)

        ImageClassificationDataset.__init__(self, dataset_dir, dataset_name, dataset_catalog=None)

        self._info = {
            "name": dataset_name,
            "dataset_dir": dataset_dir
        }
        self._num_workers = num_workers
        self._shuffle = shuffle_files
        self._preprocessed = None
        self._dataset = None
        self._train_indices = None
        self._validation_indices = None
        self._test_indices = None

        self._dataset = datasets.ImageFolder(self._dataset_dir)

        self._class_names = self._dataset.classes

        self._train_pct = 1.0
        self._val_pct = 0
        self._test_pct = 0
        self._validation_type = 'recall'
        self._train_subset = None
        self._validation_subset = None
        self._test_subset = None

    @property
    def class_names(self):
        """
        Returns the list of class names
        """
        return self._dataset.classes

    @property
    def info(self):
        """
        Returns a dictionary of information about the dataset
        """
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        """
        Returns the framework dataset object (torch.utils.data.Dataset)
        """
        return self._dataset