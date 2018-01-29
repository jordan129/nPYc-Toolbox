#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy
from scipy import interp
import statsmodels.formula.api as smf

import logging
from scipy.signal import savgol_filter
import time
import sys
import copy
from datetime import datetime, timedelta
from ..objects._msDataset import MSDataset
from ..enumerations import AssayRole, SampleType

"""
def algorithm(dataframe, args):
    
    :param dataframe: Pandas dataframe containing the following columns: Correction Batch, Run Order, SampleType, AssayRole, y
    :param args: Variable numbe
    :return: 
    return fitted
    
"""


def loess_correction(dataframe, window=11):
    """

    :param dataframe:
    :param window: Window of sequential Study
    :return:
    """

    # Raise this inside method??
    if not isinstance(window, int) & (window > 0):
        raise TypeError('window must be a positive integer')

    QCSamples = dataframe[dataframe['AssayRole'] == AssayRole.PrecisionReference & dataframe['SampleType'] == SampleType.StudyPool]

    # Convert window to fraction parameter
    frac = window / float(QCSamples.shape[0])

    # Check batches that have no SR's or SR's with 0/negative intensity and ensure no correction is done for these
    # batches and these samples are not used
    # TODO: Test pandas synthax
    # TODO: Test correction equations if all good
    # TODO: Test in a real dataset
    # Remove SR samples which have 0 or negative intensity - these are considered unusable
    QCSamples_usable = QCSamples[QCSamples['y'] > 0]

    # Proportion or criteria of numbers of SR each batch must contain
    batch_criteria = 3

    # Which batches can be corrected
    correctable_batches = QCSamples_usable.groupby('Correction Batch').count() >= batch_criteria

    # Batches and sections of the SRs in the run that can be estimated
    QCSamples_usable = QCSamples_usable[QCSamples_usable['Correction Batch'] in correctable_batches]
    # Batches and sections of the whole run that can be corrected
    correctable_data = dataframe[dataframe['Correction Batch'] in correctable_batches]

    # Hold the data indexes correctly, to assemble the fitted vector
    orig_data_index = correctable_data.index
    # Prepare the fitted array
    fitted = numpy.zeros(dataframe.shape[0])

    # Fit only using the usable QC Samples and correctible batches
    qc_model = smf.ols("y ~ I(lowess(Run Order, frac=frac)*Correction Batch + Correction Batch", data=QCSamples_usable)

    # apply the correction to the data
    # Everything else should remain at 0 - leading to non-correctible batches mantaining their
    fitted[orig_data_index] = qc_model.predict(correctable_data)

    # Fit can go negative if too many adjacent QC samples == 0; set any negative fit values to zero
    # Keep this behaviour?? - when things are corrected before?
    fitted[fitted < 0] = 0

    return fitted


def GLSRegressionPrototype(dataframe):
    """

    :param dataframe:
    :return:
    """

    QCSamples = dataframe[dataframe['AssayRole'] == AssayRole.PrecisionReference & dataframe['SampleType'] == SampleType.StudyPool]
    import statsmodels.formula.api as smf

    # sigma = numpy.linalg.diag() ## remove 0's
    qc_model = smf.ols("y ~ I(lowess(Run Order, frac=frac)*Correction Batch + Correction Batch", data=QCSamples)
    fitted = qc_model.predict()

    corrected = dataframe['FeatureIntensity'] - qc_model.predict(dataframe[:, ['RunOrder', 'Correction Batch']])

    return corrected, fitted


def _optimiseCorrection(feature, optimise):
    """
	Optimise the window function my mimising the output of `optimise(data)`
	"""
    pass



def doSavitzkyGolayCorrection(dataframe, window=11, polyOrder=3):
    """
	Fit a Savitzky-Golay curve to the data.
	"""
    # Sort the array
    #sortedRO = numpy.argsort(QCrunorder)
    #sortedRO2 = QCrunorder[sortedRO]
    #QCdataSorted = QCdata[sortedRO]

    # actually do the work
    #z = savgol_filter(QCdataSorted, window, polyOrder)

    #fit = interp(runorder, sortedRO2, z)

    #corrected = numpy.divide(data, fit)
    #corrected = numpy.multiply(corrected, numpy.median(QCdata))

    #return corrected, fit


batch_correction_algorithms = {'LOESS': loess_correction, 'GLS': GLSRegressionPrototype}