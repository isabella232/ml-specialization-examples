#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constants shared by preprocessing and modeling scripts."""


# Define the format of your input data as present in the CSV file

CSV_COLUMNS = [
 'trip_seconds',
 'air_distance',
 'pickup_latitude',
 'pickup_longitude',
 'dropoff_latitude',
 'dropoff_longitude',
 'pickup_community_area',
 'dropoff_community_area',
 'trip_start_hour',
 'trip_start_weekday',
 'trip_start_week',
 'trip_start_yearday',
 'trip_start_month'
]
CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0], [0], [0], [0], [0], [0], [0]]

LABEL_COLUMN = 'trip_seconds'