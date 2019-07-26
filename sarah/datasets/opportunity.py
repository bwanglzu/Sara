"""Opportunity dataset.

[1] Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek,
    Gerhard Tröster, Paul Lukowicz, Gerald Pirkl, David Bannach, Alois Ferscha,
    Jakob Doppler, Clemens Holzmann, Marc Kurz, Gerald Holl, Ricardo Chavarriaga,
    Hesam Sagha, Hamidreza Bayati, and José del R. Millàn.
    "Collecting complex activity data sets in highly rich networked sensor environments"
    In Seventh International Conference on Networked Sensing Systems (INSS’10),
    Kassel, Germany, 2010.
"""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests
import numpy as np

from sarah.utils import train_test_split

def load_data():
    """Download , extract & Load opportunity dataset."""
    uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases' + \
    '/00226/OpportunityUCIDataset.zip'
    opportunity_dir = Path.cwd().joinpath('OpportunityUCIDataset')
    if not opportunity_dir.exists():
        response = requests.get(uri)
        zipfile = ZipFile(BytesIO(response.content))
        zipfile.extractall()
    batches = list(opportunity_dir.glob('**/*.dat'))
    opportunity = np.empty(shape=[0, 250])
    for batch in batches:
        opportunity = np.concatenate((opportunity,np.loadtxt(batch, dtype=float)))
    X, y = opportunity[:, :243], opportunity[:, 243:]
    return train_test_split(X, y)

