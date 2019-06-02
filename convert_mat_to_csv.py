# Copyright 2019 VIA Technologies, Inc. All Rights Reserved.
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
# ==============================================================================
import sys
import os
import csv
import numpy as np
import scipy.io as sio

mat_file = sys.argv[1]
csv_file = sys.argv[2]

with open(csv_file, 'w') as csvfile:

  mat = sio.loadmat(mat_file)

  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(['relative_im_path','class','bbox_x1','bbox_y1','bbox_x2','bbox_y2','test'])

  for annotation in mat['annotations'][0]:
    test = np.squeeze(annotation['test'])
    im_path = str(np.squeeze(annotation['relative_im_path']))
    cls = np.squeeze(annotation['class'])
    x1 = np.squeeze(annotation['bbox_x1'])
    y1 = np.squeeze(annotation['bbox_y1'])
    x2 = np.squeeze(annotation['bbox_x2'])
    y2 = np.squeeze(annotation['bbox_y2'])

    csvwriter.writerow([im_path, cls, x1, y1, x2, y2, test])

