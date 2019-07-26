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
from typing import Union, Tuple

import requests
import numpy as np
from loguru import logger

from sarah.utils import train_test_split

def load_data() -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download, extract & Load opportunity dataset.

    The value will be split into train and test by :meth:`train_test_split`. Use
        (X_train, y_train), (X_test, y_test) to unpack returned value.

    :return: A tuple contains `X_train` and `y_train`.
    :return: A tuple contains `X_test` and `y_test`.
    """
    uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases' + \
    '/00226/OpportunityUCIDataset.zip'
    opportunity_dir = Path.cwd().joinpath('OpportunityUCIDataset')
    if not opportunity_dir.exists():
        logger.info('Downloading opportunity dataset...')
        response = requests.get(uri)
        logger.info('Download finished.')
        logger.info('Unzipping opportunity dataset...')
        zipfile = ZipFile(BytesIO(response.content))
        zipfile.extractall()
        logger.info('Unzip finished.')
    logger.info('Loading opportunity dataset...')
    batches = list(opportunity_dir.glob('**/*.dat'))
    opportunity = np.empty(shape=[0, 250])
    for batch in batches:
        opportunity = np.concatenate((opportunity,np.loadtxt(batch, dtype=float)))
    X, y = opportunity[:, :243], opportunity[:, 243:]
    logger.info('Load finished.')
    return train_test_split(X, y)

