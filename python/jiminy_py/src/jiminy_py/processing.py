
#!/usr/bin/env python

## @file jiminy_py/processing.py

import numpy as np
from scipy.interpolate import UnivariateSpline


def smoothing_filter(time_in, val_in, time_out=None, relabel=None, params=None):
    """
    @brief      Smoothing filter with relabeling and resampling features.

    @details    It supports evenly sampled multidimensional input signal.
                Relabeling can be used to infer the value of samples at
                time steps before and after the explicitly provided samples.
                As a reminder, relabeling is a generalization of periodicity.

    @param[in]  time_in     Time steps of the input signal (1D numpy array)
    @param[in]  val_in      Sampled values of the input signal
                            (2D numpy array: row = sample, column = time)
    @param[in]  time_out    Time steps of the output signal (1D numpy array)
    @param[in]  relabel     Relabeling matrix (identity for periodic signals)
                            Optional: Disable if omitted
    @param[in]  params      Parameters of the filter. Dictionary with keys:
                            'mixing_ratio_1': Relative time at the begining of the signal
                                            during the output signal corresponds to a
                                            linear mixing over time of the filtered and
                                            original signal. (only used if relabel is omitted)
                            'mixing_ratio_2': Relative time at the end of the signal
                                            during the output signal corresponds to a
                                            linear mixing over time of the filtered and
                                            original signal. (only used if relabel is omitted)
                            'smoothness'[0]: Smoothing factor to filter the begining of the signal
                                            (only used if relabel is omitted)
                            'smoothness'[1]: Smoothing factor to filter the end of the signal
                                            (only used if relabel is omitted)
                            'smoothness'[2]: Smoothing factor to filter the middle part of the signal

    @return     Filtered signal (2D numpy array: row = sample, column = time)
    """
    if time_out is None:
        time_out = time_in
    if params is None:
        params = dict()
        params['mixing_ratio_1'] = 0.12
        params['mixing_ratio_2'] = 0.04
        params['smoothness'] = [0.0,0.0,0.0]
        params['smoothness'][0]  = 5e-3
        params['smoothness'][1]  = 5e-3
        params['smoothness'][2]  = 3e-3

    if relabel is None:
        mix_fit    = [None,None,None]
        mix_fit[0] = lambda t: 0.5*(1+np.sin(1/params['mixing_ratio_1']*((t-time_in[0])/(time_in[-1]-time_in[0]))*np.pi-np.pi/2))
        mix_fit[1] = lambda t: 0.5*(1+np.sin(1/params['mixing_ratio_2']*((t-(1-params['mixing_ratio_2'])*time_in[-1])/(time_in[-1]-time_in[0]))*np.pi+np.pi/2))
        mix_fit[2] = lambda t: 1

        val_fit = []
        for jj in range(val_in.shape[0]):
            val_fit_jj = []
            for kk in range(len(params['smoothness'])):
                val_fit_jj.append(UnivariateSpline(time_in, val_in[jj], s=params['smoothness'][kk]))
            val_fit.append(val_fit_jj)

        time_out_mixing = [None, None, None]
        time_out_mixing_ind = [None, None, None]
        time_out_mixing_ind[0] = time_out < time_out[-1]*params['mixing_ratio_1']
        time_out_mixing[0] = time_out[time_out_mixing_ind[0]]
        time_out_mixing_ind[1] = time_out > time_out[-1]*(1-params['mixing_ratio_2'])
        time_out_mixing[1] = time_out[time_out_mixing_ind[1]]
        time_out_mixing_ind[2] = np.logical_and(np.logical_not(time_out_mixing_ind[0]), np.logical_not(time_out_mixing_ind[1]))
        time_out_mixing[2] = time_out[time_out_mixing_ind[2]]

        val_out = np.zeros((val_in.shape[0],len(time_out)))
        for jj in range(val_in.shape[0]):
            for kk in range(len(time_out_mixing)):
                val_out[jj,time_out_mixing_ind[kk]] = \
                   (1 - mix_fit[kk](time_out_mixing[kk])) * val_fit[jj][kk](time_out_mixing[kk]) + \
                        mix_fit[kk](time_out_mixing[kk])  * val_fit[jj][-1](time_out_mixing[kk])
    else:
        time_tmp   = np.concatenate([time_in[:-1]-time_in[-1],time_in,time_in[1:]+time_in[-1]])
        val_in_tmp = np.concatenate([relabel.dot(val_in[:,:-1]),val_in,relabel.dot(val_in[:,1:])], axis=1)
        val_out = np.zeros((val_in.shape[0],len(time_out)))
        for jj in range(val_in_tmp.shape[0]):
            f = UnivariateSpline(time_tmp, val_in_tmp[jj], s=params['smoothness'][-1])
            val_out[jj] = f(time_out)

    return val_out
