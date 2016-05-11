import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 



def EstimateObservationNoise(csvfile, sensor_number, start_obs, stop_obs, plot=False):
    '''
    Estimate the Gaussian noise over an observation period.  Just subtract mean and estimate variance. NaN's are omitted.
    
    Parameters
    ----------
    csvfile : filepath
        Path to the observation CSV file
    sensor_number : integer
        integer index of sensor in pandas dataframe.  
    start_obs : integer
        integer index of starting observation
    stop_obs : integer
        integer index of ending observation
    plot : bool (default true)
        If true, plot the histogram and gaussian estimator. 
    
    Returns
    ----------
    variance : Variance of observation set. 
    '''
    
    df = pd.read_pickle(csvfile)
    # Observations 
    Y = df['snowdepth_%i'%sensor_number].values
    obs = Y[start_obs:stop_obs] 
    obs = obs[~np.isnan(obs)]
    obs -= np.mean(obs)
    var = np.std(obs)**2
    bins = np.linspace(-100,100,50)
    
    if plot: 
        plt.hist(obs, bins=bins, histtype='step', normed=True, label='Sensor %i'%sensor_number)
        gaus = np.exp(-bins**2/(2*var))
        gaus = gaus / np.sum(gaus*(bins[-1]-bins[0]))
        #plt.plot(bins, gaus)
        plt.legend(frameon=False)
        plt.xlabel('Scatter [mm]')
    
    return var

def GetNumSensors(csvfile):
    '''
    Get the number of sensors in the cluster. 
    
    Parameters
    ----------
    csvfile : filepath
        Path to the observation CSV file
    
    Returns
    ----------
    nsensors : number of sensors
    '''
    df = pd.read_pickle(csvfile)
    nsensors = 0
    for col in df.columns: 
        if 'snowdepth_' in col:
            nsensors += 1
    return nsensors

def GetTimeSeries(csvfile, sensor):
    '''
    Return the sensor GetTimeSeries
    
    Parameters
    ----------
    csvfile : filepath
        Path to the observation CSV file
    sensor : sensor number to retrieve 

    Returns
    ----------
    time_series : the observation time series
    '''

    df = pd.read_pickle(csvfile)
    try: 
        Y = df['snowdepth_%i'%sensor].values
    except: 
        raise Exception('Sensor number %i not found in file'%sensor)

    return Y