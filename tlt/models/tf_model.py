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

import inspect
import os
import random
import numpy as np
import tensorflow as tf

from tlt.models.model import BaseModel
from tlt.utils.file_utils import verify_directory, validate_model_name
from tlt.utils.platform_util import PlatformUtil
from tlt.utils.types import FrameworkType, UseCaseType


class TFModel(BaseModel):
    """
    Base class to represent a TF pretrained model
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        self._model = None
        super().__init__(model_name, framework, use_case)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
        self._history = {}

    def _set_seed(self, seed):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, epochs, initial_checkpoints):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))

    def _check_optimizer_loss(self, optimizer, loss):
        if optimizer is not None and (not inspect.isclass(optimizer) or
                                      tf.keras.optimizers.Optimizer not in inspect.getmro(optimizer)):
            raise TypeError("The optimizer input must be a class (not an instance) of type "
                            "tf.keras.optimizers.Optimizer or None but found a {}. "
                            "Example: tf.keras.optimizers.SGD".format(optimizer))
        if loss is not None and (not inspect.isclass(loss) or
                                 tf.keras.losses.Loss not in inspect.getmro(loss)):
            raise TypeError("The loss input must be class (not an instance) of type "
                            "tf.keras.losses.Loss or None but found a {}. "
                            "Example: tf.keras.losses.BinaryCrossentropy".format(loss))

    def load_from_directory(self, model_dir: str):
        """
            Loads a saved model from the specified directory

            Args:
                model_dir (str): Directory with a saved_model.pb or h5py file to load

            Returns:
                None

            Raises:
                TypeError if model_dir is not a string
                NotADirectoryError if model_dir is not a directory
                IOError for an invalid model file
        """
        # Verify that the model directory exists
        verify_directory(model_dir, require_directory_exists=True)

        self._model = tf.keras.models.load_model(model_dir)
        self._model.summary(print_fn=print)

    def set_auto_mixed_precision(self, enable_auto_mixed_precision):
        """
        Enable auto mixed precision for training. Mixed precision uses both 16-bit and 32-bit floating point types to
        make training run faster and use less memory. If enable_auto_mixed_precision is set to None, auto mixed
        precision will be enabled when running with Intel fourth generation Xeon processors, and disabled for other
        platforms.
        """
        if enable_auto_mixed_precision is not None and not isinstance(enable_auto_mixed_precision, bool):
            raise TypeError("Invalid type for enable_auto_mixed_precision. Expected None or a bool.")

        # Get the TF version
        tf_major_version = 0
        tf_minor_version = 0
        if tf.version.VERSION is not None and '.' in tf.version.VERSION:
            tf_version_list = tf.version.VERSION.split('.')
            if len(tf_version_list) > 1:
                tf_major_version = int(tf_version_list[0])
                tf_minor_version = int(tf_version_list[1])

        auto_mixed_precision_supported = (tf_major_version == 2 and tf_minor_version >= 9) or tf_major_version > 2

        if enable_auto_mixed_precision is None:
            # Determine whether or not to enable this based on the CPU type
            try:
                # Only enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil(args=None).cpu_type == 'SPR'
            except Exception as e:
                if auto_mixed_precision_supported:
                    print("Unable to determine the CPU type:", str(e))
                enable_auto_mixed_precision = False
        elif not auto_mixed_precision_supported:
            print("Warning: Auto mixed precision requires TensorFlow 2.9.0 or later (found {}).".format(
                tf.version.VERSION))

        if auto_mixed_precision_supported:
            if enable_auto_mixed_precision:
                print("Enabling auto_mixed_precision_mkl")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision_mkl': enable_auto_mixed_precision})

    def export(self, output_dir):
        """
           Exports a trained model as a saved_model.pb file. The file will be written to the output directory in a
           directory with the model's name, and a unique numbered directory (compatible with TF serving). The directory
           number will increment each time the model is exported.

           Args:
               output_dir (str): A writeable output directory.

           Returns:
               The path to the numbered saved model directory

           Raises:
               TypeError if the output_dir is not a string
               FileExistsError the specified output directory already exists as a file
               ValueError if the mode has not been loaded or trained yet
        """
        if self._model:
            # Save the model in a format that can be served
            verify_directory(output_dir)
            val_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, val_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")

            self._model.save(saved_model_dir)
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been loaded or trained yet")