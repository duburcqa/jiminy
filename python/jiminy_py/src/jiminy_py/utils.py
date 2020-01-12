#!/usr/bin/env python

## @file jiminy_py/utils.py

import jiminy


def get_jiminy_model(trajectory_data):
    jiminy_model = trajectory_data["jiminy_model"]
    if (jiminy_model is None):
        urdf_path = trajectory_data["urdf"]
        has_freeflyer = trajectory_data["has_freeflyer"]
        jiminy_model = jiminy.Model()
        jiminy_model.initialize(urdf_path, [], [], has_freeflyer)
    return jiminy_model