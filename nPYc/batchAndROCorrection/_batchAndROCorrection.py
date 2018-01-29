#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
import numpy
import scipy
from scipy import interp
from scipy.signal import savgol_filter
import statsmodels.api as sm
import _batchCorrectionAlgorithms
import tempfile
import os
import shutil
import tempfile

# Set up multiprocessing enviroment
import multiprocessing

from joblib import load, dump, Parallel, delayed

lowess = sm.nonparametric.lowess
import logging
from scipy.signal import savgol_filter
import time
import sys
import copy
from datetime import datetime, timedelta
from ..objects._msDataset import MSDataset
from ..enumerations import AssayRole, SampleType
import pdb
import pandas


def correctMSdataset(data, method='LOWESS', align='median', parallelise=True,
                                excludeFailures=True, **kwargs):
    """
    Conduct run-order correction and batch alignment on the :py:class:`~nPYc.objects.MSDataset` instance *data*, returning a new instance with corrected intensity values.

    Sample are seperated into batches acording to the *'Correction Batch'* column in *data.sampleMetadata*.

    :param data: MSDataset object with measurements to be corrected
    :type data: MSDataset
    :param int window: When calculating trends, use a consider this many reference samples, centred on the current position
    :param str method: Correction method, one of 'LOWESS' (default), 'SavitzkyGolay' or None for no correction
    :param str align: Average calculation of batch and feature intensity for correction, one of 'median' (default) or 'mean'
    :param bool parallelise: If ``True``, use multiple cores
    :param bool excludeFailures: If ``True``, remove features where a correct fit could not be calculated from the dataset
    :return: Duplicate of *data*, with run-order correction applied
    :rtype: MSDataset
    """
    import copy

    # Check inputs
    if not isinstance(data, MSDataset):
        raise TypeError("data must be a MSDataset instance")

    if method is not None:
        if not isinstance(method, str) & (method in _batchCorrectionAlgorithms.batch_correction_algorithms.keys()):
            raise ValueError('method must be == LOWESS or SavitzkyGolay')
    if not isinstance(align, str) & (align in {'mean', 'median'}):
        raise ValueError('align must be == mean or median')
    if not isinstance(parallelise, bool):
        raise TypeError("parallelise must be a boolean")
    if not isinstance(excludeFailures, bool):
        raise TypeError("excludeFailures must be a boolean")

    # Assemble a data frame with the necessary information
    correction_dataframe = data.sampleMetadata.loc['Run Order', 'SampleType', 'AssayRole', 'Correction Batch'].values

    correctedP = _batchCorrection(data.intensityData, correction_dataframe, method=method, align=align,
                                      parallelise=parallelise, **kwargs)

    correctedData = copy.deepcopy(data)
    correctedData.intensityData = correctedP[0]
    correctedData.fit = correctedP[1]
    correctedData.Attributes['Log'].append([datetime.now(), 'Batch and run order correction applied'])

    return correctedData


def _batchCorrection(data, dataframe, method='LOWESS',
                         parallelise=True, savePlots=False, **kwargs):
    """
	Conduct run-order correction and batch alignment.

	:param data: Raw *n* × *m* numpy array of measurements to be corrected
	:type data: numpy.array
	:param runOrder: *n* item list of order of analysis
	:type runOrder: numpy.series
	:param referenceSamples: *n* element boolean array indicating reference samples to base the correction on
	:type referenceSamples: numpy.series
	:param batchList: *n* item list of correction batch, defines sample groupings into discrete batches for correction
	:type batchList: numpy.series
	:param int window: When calculating trends, use a consider this many reference samples, centred on the current position
	:param str method: Correction method, one of 'LOWESS' (default), 'SavitzkyGolay' or None for no correction
	:param str align: Average calculation of batch and feature intensity for correction, one of 'median' (default) or 'mean'
	"""
    # Validate inputs
    if not isinstance(data, numpy.ndarray):
        raise TypeError('data must be a numpy array')
    if method is not None:
        if not isinstance(method, str) & (method in _batchCorrectionAlgorithms.batch_correction_algorithms.keys()):
            raise ValueError('method must be == LOWESS or SavitzkyGolay')
    if not isinstance(parallelise, bool):
        raise TypeError('parallelise must be True or False')
    if not isinstance(savePlots, bool):
        raise TypeError('savePlots must be True or False')

    # Store paramaters in a dict to avoid arg lists going out of control
    parameters = dict()
    parameters['method'] = method
    parameters['align'] = align
    # Parse kwargs
    for key, value in kwargs.items():
        parameters[key] = value

    if parallelise:
        # Set up multiprocessing enviroment
        import multiprocessing

        # Generate an index and set up pool
        # Use one less workers than CPU cores
        if multiprocessing.cpu_count() - 1 <= 0:
            cores = 1
        else:
            cores = multiprocessing.cpu_count() - 1

        pool = multiprocessing.Pool(processes=cores)

        instances = range(0, cores)

        # Break features into no cores chunks
        featureIndex = _chunkMatrix(range(0, data.shape[1]), cores)

        # Ship the parallel calculations
        results2 = [pool.apply_async(_batchCorrectionWorker,
                                     args=(data, dataframe, featureIndex, parameters, w))
                    for w in instances]

        results2 = [p.get(None) for p in results2]

        results = list()
        # Unpack results
        for instanceOutput in results2:
            for item in instanceOutput:
                results.append(item)

        # Shut down the pool
        pool.close()

    else:
        # Just run it
        # Iterate over features in one batch and correct them
        results = _batchCorrectionWorker(data, dataframe,
                                         range(0, data.shape[1]),
                                         parameters, 0)
    correctedData = numpy.empty_like(data)
    fits = numpy.empty_like(data)

    # Extract return values from tuple
    for (w, feature, fit) in results:
        correctedData[:, w] = feature
        fits[:, w] = fit

    return correctedData, fits


def _batchCorrectionWorker(data, dataframe, featureIndex, parameters, w):
    """
    Breaks the dataset into batches to be corrected together and handles the feature iteration
    :param data:
    :param dataframe:
    :param featureIndex:
    :param parameters:
    :param w:
    :return:
    """
    # Check if we have a list of lists, or just one list:
    if isinstance(featureIndex[0], range):
        featureList = featureIndex[w]
    else:
        featureList = range(0, len(featureIndex))

    # Detect which samples are QC
    qcSamples = dataframe[dataframe['AssayRole'] == AssayRole.PrecisionReference & dataframe['SampleType'] == SampleType.StudyPool]
    # Check for the samples which are SR but not meant to be included in correction
    # (conditioning, etc other stuff).

    # add results to this list:
    results = list()

    # Loop over all elements in featureList
    for i in featureList:

        # Create a matrix to be used with `nonlocal` to store fits
        try:
            feature = copy.deepcopy(data[:, i])
        except IndexError:
            feature = copy.deepcopy(data)
        fit = numpy.empty_like(feature)
        fit.fill(numpy.nan)

        # Generate a copy of the dataframe with the intensity features for this dataset
        feature_dataframe = dataframe.assign(y=feature)

        fitted = _batchCorrectionAlgorithms.batch_correction_algorithms[parameters['method']](feature_dataframe, **parameters)
        feature = feature / fitted

        exclude = list()
        results.append((i, feature, fit, exclude))

    return results

def optimiseCorrection(feature, optimise):
    """
	Optimise the window function my mimising the output of `optimise(data)`
	"""
    pass


##
# Adapted from http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
##
def _chunkMatrix(seq, num):
    avg = round(len(seq) / float(num))
    out = []
    last = 0.0

    for i in range(0, num - 1):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    out.append(seq[int(last):max(seq) + 1])

    return out
