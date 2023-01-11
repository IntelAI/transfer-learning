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
import math
import pytest
import shutil
import tempfile
from numpy.testing import assert_array_equal
from PIL import Image

from tlt.datasets.dataset_factory import get_dataset, load_dataset

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset  # noqa: E501
except ModuleNotFoundError:
    print("WARNING: Unable to import TorchvisionImageClassificationDataset. Torch may not be installed")

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset import PyTorchCustomImageClassificationDataset  # noqa: E501
except ModuleNotFoundError:
    print("WARNING: Unable to import PyTorchCustomImageClassificationDataset. Torch may not be installed")

try:
    from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset
except ModuleNotFoundError:
    print("WARNING: Unable to import HFTextClassificationDataset. HuggingFace's 'tranformers' API may not be installed \
            in the current env")


@pytest.mark.pytorch
def test_torchvision_subset():
    """
    Checks that a torchvision test subset can be loaded
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision', split=["test"])
    assert type(data) == TorchvisionImageClassificationDataset
    assert len(data.dataset) < 50000


@pytest.mark.pytorch
def test_defined_split():
    """
    Checks that dataset can be loaded into train and test subsets based on torchvision splits and then
    re-partitioned with shuffle-split
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10',
                       'torchvision', split=['train', 'test'])
    assert len(data.dataset) == 60000
    assert len(data.train_subset) == 50000
    assert len(data.test_subset) == 10000
    assert data.validation_subset is None
    assert data._train_indices == range(50000)
    assert data._test_indices == range(50000, 60000)
    assert data._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    data.shuffle_split(.6, .2, .2, seed=10)
    assert len(data.train_subset) == 36000
    assert len(data.validation_subset) == 12000
    assert len(data.test_subset) == 12000
    assert data._validation_type == 'shuffle_split'


@pytest.mark.pytorch
def test_shuffle_split():
    """
    Checks that dataset can be split into train, validation, and test subsets
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision')
    data.shuffle_split(seed=10)
    assert len(data.train_subset) == 37500
    assert len(data.validation_subset) == 12500
    assert data.test_subset is None
    assert data._validation_type == 'shuffle_split'


@pytest.mark.pytorch
def test_shuffle_split_deterministic_tv():
    """
    Checks that dataset can be split into train, validation, and test subsets in a way that is reproducible
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'DTD', 'torchvision', split=['test'])
    data.preprocess(224, 128)
    data.shuffle_split(seed=10)

    data2 = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'DTD', 'torchvision', split=['test'])
    data2.preprocess(224, 128)
    data2.shuffle_split(seed=10)

    for i in range(3):
        image_1, label_1 = data.get_batch()
        image_2, label_2 = data2.get_batch()
        assert_array_equal(image_1, image_2)
        assert_array_equal(label_1, label_2)


@pytest.mark.pytorch
def test_shuffle_split_deterministic_custom():
    """
    Checks that custom datasets can be split into train, validation, and test subsets in a way that is reproducible
    """
    dataset_dir = '/tmp/data'
    class_names = ['foo', 'bar']
    seed = 10
    image_size = 224
    batch_size = 1
    ic_dataset1 = None
    ic_dataset2 = None
    try:
        ic_dataset1 = ImageClassificationDatasetForTest(dataset_dir, None, None, class_names)
        tlt_dataset1 = ic_dataset1.tlt_dataset
        tlt_dataset1.preprocess(image_size, batch_size)
        tlt_dataset1.shuffle_split(seed=seed)

        ic_dataset2 = ImageClassificationDatasetForTest(dataset_dir, None, None, class_names)
        tlt_dataset2 = ic_dataset2.tlt_dataset
        tlt_dataset2.preprocess(image_size, batch_size)
        tlt_dataset2.shuffle_split(seed=seed)

        for i in range(10):
            image_1, label_1 = tlt_dataset1.get_batch()
            image_2, label_2 = tlt_dataset2.get_batch()
            assert_array_equal(image_1, image_2)
            assert_array_equal(label_1, label_2)
    finally:
        if ic_dataset1:
            ic_dataset1.cleanup()
        if ic_dataset2:
            ic_dataset2.cleanup()


@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names,batch_size',
                         [['/tmp/data', 'DTD', 'torchvision', None, 32],
                          ['/tmp/data', 'DTD', 'torchvision', None, 1],
                          ['/tmp/data', None, None, ['foo', 'bar'], 8],
                          ['/tmp/data', None, None, ['foo', 'bar'], 1]])
def test_batching(dataset_dir, dataset_name, dataset_catalog, class_names, batch_size):
    """
    Checks that dataset can be batched with valid positive integer values
    """
    ic_dataset = ImageClassificationDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlt_dataset = ic_dataset.tlt_dataset

        tlt_dataset.preprocess(224, batch_size)
        assert len(tlt_dataset.get_batch()[0]) == batch_size
    finally:
        ic_dataset.cleanup()


@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names',
                         [['/tmp/data', 'DTD', 'torchvision', None],
                          ['/tmp/data', None, None, ['foo', 'bar']]])
def test_batching_error(dataset_dir, dataset_name, dataset_catalog, class_names):
    """
    Checks that preprocessing cannot be run twice
    """
    ic_dataset = ImageClassificationDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlt_dataset = ic_dataset.tlt_dataset
        tlt_dataset.preprocess(224, 1)
        with pytest.raises(Exception) as e:
            tlt_dataset.preprocess(256, 32)
        assert 'Data has already been preprocessed: {}'.\
            format(tlt_dataset._preprocessed) == str(e.value)
    finally:
        ic_dataset.cleanup()


class ImageClassificationDatasetForTest:
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for image classification datasets (either from torchvision or custom).

        For a custom dataset, provide a dataset dir and class names. A temporary directory will be created with
        dummy folders for the specified class names and 50 images in each folder. The dataset factory will be used to
        load the custom dataset from the dataset directory.

        For an image classification dataset from a catalog, provide the dataset_dir, dataset_name, and dataset_catalog.
        The dataset factory will be used to load the specified dataset.
        """
        use_case = 'image_classification'
        framework = 'pytorch'

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlt_dataset = get_dataset(dataset_dir, use_case, framework, dataset_name, dataset_catalog)
        elif class_names:
            self._dataset_catalog = "custom"
            dataset_dir = tempfile.mkdtemp(dir=dataset_dir)
            if not isinstance(class_names, list):
                raise TypeError("class_names needs to be a list")

            for dir_name in class_names:
                image_class_dir = os.path.join(dataset_dir, dir_name)
                os.makedirs(image_class_dir)
                for n in range(50):
                    img = Image.new(mode='RGB', size=(24, 24))
                    img.save(os.path.join(image_class_dir, 'img_{}.jpg'.format(n)))

            self._tlt_dataset = load_dataset(dataset_dir, use_case, framework)

        self._dataset_dir = dataset_dir

    @property
    def tlt_dataset(self):
        """
        Returns the tlt dataset object
        """
        return self._tlt_dataset

    def cleanup(self):
        """
        Clean up - remove temp files that were created for custom datasets
        """
        if self._dataset_catalog == "custom":
            print("Deleting temp directory:", self._dataset_dir)
            shutil.rmtree(self._dataset_dir)
            # TODO: Should we delete torchvision directories too?


# Metadata about torchvision datasets
torchvision_metadata = {
    'CIFAR10': {
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'size': 50000
    }
}

# Dataset parameters used to define datasets that will be initialized and tested using TestImageClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, and class_names, which map to the constructor
# parameters for ImageClassificationDatasetForTest, which initializes the datasets using the dataset factory.
dataset_params = [("/tmp/data", "CIFAR10", "torchvision", None),
                  ("/tmp/data", None, None, ["a", "b", "c"])]


@pytest.fixture(scope="class", params=dataset_params)
def image_classification_data(request):
    params = request.param

    ic_dataset = ImageClassificationDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, dataset_classes = params

    def cleanup():
        ic_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlt dataset along with metadata that tests might need
    return (ic_dataset.tlt_dataset, dataset_name, dataset_classes)


@pytest.mark.pytorch
class TestImageClassificationDataset:
    """
    This class contains image classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the dataset defined in the dataset_params list.
    """

    @pytest.mark.pytorch
    def test_class_names_and_size(self, image_classification_data):
        """
        Verify the class type, dataset class names, and dataset length after initializaion
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data

        if dataset_name is None:
            assert type(tlt_dataset) == PyTorchCustomImageClassificationDataset
            assert len(tlt_dataset.class_names) == len(dataset_classes)
            assert len(tlt_dataset.dataset) == len(dataset_classes) * 50
        else:
            assert type(tlt_dataset) == TorchvisionImageClassificationDataset
            assert len(tlt_dataset.class_names) == len(torchvision_metadata[dataset_name]['class_names'])
            assert len(tlt_dataset.dataset) == torchvision_metadata[dataset_name]['size']

    @pytest.mark.pytorch
    @pytest.mark.parametrize('batch_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_batch_sizes(self, batch_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid batch size is passed
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess(224, batch_size)

    @pytest.mark.pytorch
    @pytest.mark.parametrize('image_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_image_size(self, image_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid image size is passed
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess(image_size, batch_size=8)

    @pytest.mark.pytorch
    def test_preprocessing(self, image_classification_data):
        """
        Checks that dataset can be preprocessed only once
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data
        tlt_dataset.preprocess(224, 8)
        preprocessing_inputs = {'image_size': 224, 'batch_size': 8}
        assert tlt_dataset._preprocessed == preprocessing_inputs
        # Trying to preprocess again should throw an exception
        with pytest.raises(Exception) as e:
            tlt_dataset.preprocess(324, 32)
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == str(e.value)
        print(tlt_dataset.info)

    @pytest.mark.pytorch
    def test_shuffle_split_errors(self, image_classification_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data

        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=1, val_pct=0)
        assert 'Percentage arguments must be floats.' == str(e.value)

    @pytest.mark.pytorch
    def test_shuffle_split(self, image_classification_data):
        """
        Checks that dataset can be split into train, validation, and test subsets
        """
        tlt_dataset, dataset_name, dataset_classes = image_classification_data

        # Before the shuffle split, validation type should be recall
        assert 'recall' == tlt_dataset._validation_type

        # Perform shuffle split with default percentages
        tlt_dataset.shuffle_split(seed=10)
        default_train_pct = 0.75
        default_val_pct = 0.25

        # Get the full dataset size
        ds_size = torchvision_metadata[dataset_name]['size'] if dataset_name else len(dataset_classes) * 50

        # Divide by the batch size that was used to preprocess earlier
        ds_size = ds_size / tlt_dataset.info['preprocessing_info']['batch_size']

        # The PyTorch loaders are what gets batched and they can be off by 1 from the floor value
        assert math.floor(
            ds_size * default_train_pct) <= len(tlt_dataset.train_loader) <= math.ceil(ds_size * default_train_pct)
        assert math.floor(
            ds_size * default_val_pct) <= len(tlt_dataset.validation_loader) <= math.ceil(ds_size * default_val_pct)
        assert tlt_dataset.test_loader is None
        assert tlt_dataset._validation_type == 'shuffle_split'

# =======================================================================================

# Testing for Text classification use case


hf_metadata = {
    'imdb': {
        'class_names': ['neg', 'pos'],
        'size': 25000
    }
}


class TextClassificationDatasetForTest:
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for text classification datasets from Hugging Face.

        For a text classification dataset from Hugging Face catalog, provide the dataset_dir, dataset_name, and \
        dataset_catalog. The dataset factory will be used to load the specified dataset.
        """
        use_case = 'text_classification'
        framework = 'pytorch'

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlt_dataset = get_dataset(dataset_dir, use_case, framework, dataset_name, dataset_catalog)

        self._dataset_dir = dataset_dir

    @property
    def tlt_dataset(self):
        """
        Returns the tlt dataset object
        """
        return self._tlt_dataset


# Dataset parameters used to define datasets that will be initialized and tested using TestTextClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, dataset_classes which map to the constructor
# parameters for TextClassificationDatasetForTest, which initializes the dataset using the dataset factory.
dataset_params = [("/tmp/data", "imdb", "huggingface", ['neg', 'pos'])]


@pytest.fixture(scope="class", params=dataset_params)
def text_classification_data(request):
    params = request.param

    tc_dataset = TextClassificationDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, class_names = params

    # Return the tlt dataset along with metadata that tests might need
    return (tc_dataset.tlt_dataset, dataset_dir, dataset_name, dataset_catalog, class_names)


@pytest.mark.pytorch
class TestTextClassificationDataset:
    """
    This class contains text classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the datasets defined in the dataset_params list.
    """

    @pytest.mark.pytorch
    def test_tlt_dataset(self, text_classification_data):
        """
        Tests whether a matching TLT dataset object is returned
        """
        tlt_dataset, _, _, _, _ = text_classification_data
        assert type(tlt_dataset) == HFTextClassificationDataset

    @pytest.mark.pytorch
    def test_class_names_and_size(self, text_classification_data):
        """
        Verify the class type, dataset class names, and dataset length after initializaion
        """
        tlt_dataset, _, dataset_name, _, class_names = text_classification_data
        assert tlt_dataset.class_names == class_names
        assert len(tlt_dataset.dataset) == hf_metadata[dataset_name]['size']

    @pytest.mark.pytorch
    @pytest.mark.parametrize('batch_size',
                             ['foo',  # A string
                              -17,  # A negative int
                              20.5,  # A float
                              ])
    def test_invalid_batch_size_type(self, batch_size, text_classification_data):
        """
        Ensures that a ValueError is raised when an invalid batch size type is passed
        """
        tlt_dataset, _, _, _, _ = text_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess('', batch_size)

    @pytest.mark.pytorch
    def test_shuffle_split_errors(self, text_classification_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlt_dataset, _, _, _, _ = text_classification_data
        with pytest.raises(ValueError) as sum_err_message:
            tlt_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)

        with pytest.raises(ValueError) as float_err_message:
            tlt_dataset.shuffle_split(train_pct=1, val_pct=0)

        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(sum_err_message.value)
        assert 'Percentage arguments must be floats.' == str(float_err_message.value)