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
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict


def copy_files_src_to_tgt(samples, fns_dict, src_folder, tgt_folder):
    for sample in samples:
        files_to_copy = fns_dict.get(sample)
        for _file in files_to_copy:
            src_fn = os.path.join(src_folder, _file)
            tgt_fn = os.path.join(tgt_folder, _file)
            shutil.copy2(src_fn, tgt_fn)


def split_images(src_folder, tgt_folder):
    labels = os.listdir(src_folder)
    print("Number of labels = ", len(labels))
    print("Labels are: \n", labels)
    for label in labels:
        fns = os.listdir(os.path.join(src_folder, label))
        fns.sort()
        fns_root = ['_'.join(x.split('_')[:2]) for x in fns]
        # Convert list of tuples to dictionary value lists
        print("\nCreating default dict for stratifying the subject in {}.".format(label))
        fns_dict = defaultdict(list)
        for i, j in zip(fns_root, fns):
            fns_dict[i].append(j)
        train_samples, test_samples = train_test_split(list(fns_dict.keys()), test_size=0.2, random_state=100)

        src_dir = os.path.join(src_folder, label)
        tgt_dir = os.path.join(tgt_folder, 'train', label)
        os.makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(train_samples, fns_dict, src_dir, tgt_dir)

        tgt_dir = os.path.join(tgt_folder, 'test', label)
        os.makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(test_samples, fns_dict, src_dir, tgt_dir)

        print("Done splitting the files for label = {}\n".format(label))
    print("Done splitting the data. Output data is here: ", tgt_folder)


def get_subject_id(image_name):
    image_name = image_name.split("/")[-1]
    patient_id = "".join(image_name.split("_")[:2])[1:]
    return patient_id


def create_patient_id_list(image_data_folder, folder):
    folder_pth = os.path.join(folder, image_data_folder)
    patient_id_list = []
    for fldr in os.listdir(folder_pth):
        for f in os.listdir(os.path.join(folder_pth, fldr)):
            patient_id_list.append(get_subject_id(f))

    return np.unique(patient_id_list)


def read_annotation_file(
    folder,
    file_name,
    label_column,
    data_column,
    patient_id,
    patient_id_list,
    image_data_folder
):
    df_path = os.path.join(folder, file_name)
    df = pd.read_csv(df_path)
    label_map, reverse_label_map = label2map(df, label_column)

    if patient_id_list is not None:
        df = df[df[patient_id].isin(patient_id_list)]
    else:
        image_name_list = []
        for label in os.listdir(image_data_folder):
            image_name_list.extend(os.listdir(os.path.join(image_data_folder, label)))
        df = df[df[patient_id].isin(np.unique([get_subject_id(i) for i in image_name_list]))]

    df_new = pd.DataFrame(columns=[label_column, data_column, patient_id])
    for i in df[patient_id].unique():
        annotation = " ".join(df[df[patient_id].isin([i])][data_column].to_list())
        temp_labels = df[df[patient_id] == i][label_column].unique()
        if len(temp_labels) == 1:
            df_new.loc[len(df_new)] = [temp_labels[0], annotation, i]
        else:
            if patient_id_list is not None:
                # this is the case only shows for inference
                # label assigne as a place holder
                df_new.loc[len(df_new)] = ["Normal", annotation, i]
            else:
                Warning("Conflict in labelling ....")

    return df_new, label_map, reverse_label_map


def label2map(df, label_column):
    label_map, reverse_label_map = {}, {}
    for i, v in enumerate(df[label_column].unique().tolist()):
        label_map[v] = i
        reverse_label_map[i] = v

    return label_map, reverse_label_map


def create_train_test_set(df, patient_id, patient_id_list):
    train_label, test_label = train_test_split(
        patient_id_list, test_size=0.33, random_state=42
    )

    df_test = df[df[patient_id].isin(test_label)]
    df_train = df[df[patient_id].isin(train_label)]

    return df_train, df_test


def split_annotation(folder, file_name, image_data_folder):
    label_column = "label"
    data_column = "symptoms"
    patient_id = "Patient_ID"
    patient_id_list = None

    df, label_map, reverse_label_map = read_annotation_file(
        folder,
        file_name,
        label_column,
        data_column,
        patient_id,
        patient_id_list,
        image_data_folder
    )

    patient_id_list = create_patient_id_list(image_data_folder, folder)
    df_train, df_test = create_train_test_set(df, patient_id, patient_id_list)

    return df_train
