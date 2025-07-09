# ca: connectivity analysis

from numpy import correlate, average, angle, mean, sign, exp, zeros, abs, unwrap, corrcoef, conjugate, pi, real, asarray, less, array, nan, nanmin, isnan, vstack, unique
from scipy.signal import hilbert, csd, butter, lfilter, stft, argrelextrema
from statsmodels.tsa.stattools import grangercausalitytests

def connectivity_analysis(epochs, method, fs=None, imaginary=False, dtail=False):
    if epochs.ndim == 3: # If epochs is a 3d array (set of epochs)...
        # Create an empty array of size number_of_epochs x number_of_channels x number_of_channels
        result = zeros((epochs.shape[0], epochs.shape[1], epochs.shape[1]))
        for epoch_number, epoch in enumerate(epochs):    
            # Combine all pairs of channels 
            channels = list(range(len(epoch)))
            # for PAC the order of the channel pair matters for the result, so we need to compute all pairs
            if method == PAC: 
                for channel_a in range(len(channels)):                             
                    for channel_b in range(channel_a, len(channels)): 
                        result[epoch_number, channel_a, channel_b] = method( epoch[channel_a], epoch[channel_b], fs=fs,  imaginary=imaginary)
                        result[epoch_number, channel_b, channel_a] = method( epoch[channel_b], epoch[channel_a], fs=fs,  imaginary=imaginary)
            if method == correlation:
                result[epoch_number] = method(epoch)
            else:
                for channel_a in range(len(channels)):                             
                    for channel_b in range(channel_a, len(channels)): 
                        value = method( epoch[channel_a], epoch[channel_b], fs=fs,  imaginary=imaginary)
                        result[epoch_number, channel_a, channel_b] = value
                        result[epoch_number, channel_b, channel_a] = value
    elif epochs.ndim == 2: # if it is just one epoch
        epoch = epochs
        channels = list(range(len(epoch)))
        result = zeros((len(channels),len(channels)))
        if method == PAC: 
            for channel_a in range(len(channels)):                             
                for channel_b in range(channel_a, len(channels)): 
                    result[channel_a, channel_b] = method( epoch[channel_a], epoch[channel_b], fs=fs,  imaginary=imaginary)
                    result[channel_b, channel_a] = method( epoch[channel_b], epoch[channel_a], fs=fs,  imaginary=imaginary)
        if method == correlation:
            result = method(epoch)
        else:
            for channel_a in range(len(channels)):                             
                for channel_b in range(channel_a, len(channels)): 
                    value = method( epoch[channel_a], epoch[channel_b], fs=fs,  imaginary=imaginary)
                    result[channel_a, channel_b] = value
                    result[channel_b, channel_a] = value
    return result

def nearest_value_index(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return idx

def band_pass_filter(input_signal, sampling_frequency, frequency_band, order=5):
    # Initialize frequencies
    frequencies = {"delta": [1,4],
                   "theta": [4,8],
                   "alpha": [8,13],
                   "beta": [13,35],
                   "low_gamma": [35,60],
                   "high_gamma": [60,140]}
    # Associate the name of the selected frequency to the range it covers and that will be the low and upper margings of the filter taking into account nyquist
    nyq = 0.5 * sampling_frequency
    low = frequencies[frequency_band][0] / nyq
    high = frequencies[frequency_band][1] / nyq
    if high > 1: high = 0.9999
    # Design the filter
    b, a = butter(order, [low,high], btype='band', analog=False)
    #Apply the filter to the signal
    filtered_signal = lfilter(b, a, input_signal)
    return filtered_signal


def phase_lock(signal1, signal2, **kwars):
    
    '''Phase Locking Value (PLV) of two signals, signal1 and signal2
    
       Returns the phase locking value, that comprehends the range [0,1]
       Signals that are more synchronized in phase are more closer to 1. 
       
       NOTE: To properly obtain the PLV of two signals, they have to be bandpass filtered beforehand!!
    '''
    #Hilbert transformations -> obtains the analytic signal (complex number)
    sig1_hil = hilbert(signal1)  
    sig2_hil = hilbert(signal2)
    #Argument of the complex number, instant phase angle
    phase1 = angle(sig1_hil)   
    phase2 = angle(sig2_hil)
    #Phases Difference
    phase_dif = phase1-phase2      
    #Phase Locking value                       
    plv = abs(mean(exp(complex(0,1)*phase_dif)))    
    return plv

def phase_lag(signal1, signal2, **kwars):
    
    '''Phase Lag Index (PLI) of two signals, signal1 and signal2
    
       Returns the phase locking value, that comprehends the range [0,1]
       A PLI of zero indicates either no coupling or coupling with a phase difference centered around 0 mod π.
       A PLI of 1 indicates perfect phase locking at a value of Δϕ different from 0 mod π. 
       The stronger this nonzero phase locking is, the larger PLI will be. Note that PLI does no longer indicate,
       which of the two signals is leading in phase. Whenever needed, however, this information can be easily recovered, 
       for instance, by omitting the absolute value when computing PLI. 
    '''
    #Hilbert transformations -> obtains the analytic signal (complex number)
    sig1_hil = hilbert(signal1)                 
    sig2_hil = hilbert(signal2)
    #Argument of the complex number, instant phase angle
    phase1 = angle(sig1_hil)                 
    phase2 = angle(sig2_hil)
    #Phases Difference
    phase_dif = phase1-phase2   
    # Phase Lag Index                
    pli = abs(mean(sign(phase_dif)))      
    return pli

def spectral_coherence(signal1, signal2, fs, imaginary):
    
    # Epsilon to avoid division by zero and therefore NaNs
    epsilon = 1e-10 
    
    # cross power spectral density (csd), Pxy, using Welch’s method.
    Pxy = csd(signal1, signal2, fs=fs, scaling='spectrum')[1] 
    Pxx = csd(signal1, signal1, fs=fs, scaling='spectrum')[1]
    Pyy = csd(signal2,signal2,fs=fs, scaling='spectrum')[1]
    # imaginary coherence
    if imaginary: return average((Pxy.imag)**2/(Pxx*Pyy + epsilon))  
    # coherence
    elif not imaginary: return average(abs(Pxy)**2/(Pxx*Pyy + epsilon)) 

def spectral_coherence_real(signal1, signal2, fs, **kwars):
    return spectral_coherence(signal1, signal2, fs, imaginary=False)

def spectral_coherence_imag(signal1, signal2, fs, **kwars):
    return spectral_coherence(signal1, signal2, fs, imaginary=True)

def cross_correlation(signal1, signal2, **kwars):                                                    
    return correlate(signal1, signal2, mode="valid")

def correlation(epoch, **kwars):
    correlation_matrix = corrcoef(epoch);
    return correlation_matrix

def PEC(nse,n):
    pass

def granger_causality(signal1, signal2, maxlag=50, **kwars):
    # Check that none of the signals are constant, if so, granger causality is 0. 
    if (len(signal1) == 0 or len(signal2) == 0) or (len(unique(signal1)) == 1) or (len(unique(signal2)) == 1):
        return 0
    else:
        x = vstack([signal2, signal1]).transpose()
        test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)
        min_p_value = min(round(test_result[i+1][0]['ssr_chi2test'][0], 4) for i in range(maxlag) if i+1 in test_result)
        return min_p_value

    
def epileptogenicity_index(epoch, fs, **kwars):
    """ Epileptogenicity Index (Bartolomei 2008) """
    # Short time fourier transform of the time series x[n]
    frequencies, time_segments, X_w = stft(epoch,fs=fs, nperseg=epoch.shape[-1]/4)
    # Energy spectral density
    energy_spectral_density = real(X_w * conjugate(X_w))/(2*pi) # Take the real part because after multiplying X(w) by its conjugate all the complex parts become 0.
    
    energy_theta = energy_spectral_density[:,nearest_value_index(frequencies, 4):nearest_value_index(frequencies, 8),:].sum(axis=1)
    energy_alpha = energy_spectral_density[:,nearest_value_index(frequencies, 8):nearest_value_index(frequencies, 13),:].sum(axis=1)
    energy_beta = energy_spectral_density[:,nearest_value_index(frequencies, 13):nearest_value_index(frequencies, 35),:].sum(axis=1)
    energy_gamma = energy_spectral_density[:,nearest_value_index(frequencies, 35):,:].sum(axis=1)
    
    # Energy ratio
    energy_ratio = (energy_beta + energy_gamma)/(energy_theta + energy_alpha)
    energy_ratio_mean = energy_ratio.mean(axis=1)
    
    # Detection of rapid discharges
    lamb = .5 # lambda 
  
    Nd = []
    for channel in range(energy_ratio.shape[0]):
        U_temp = [] # vector of U_N
        # Na_temp = nan # alarm time
        Nd_temp = nan # last local minima before alarm time
        reinit = False
        for step in range(energy_ratio.shape[1]):
            U_n =  energy_ratio[channel,:step+1].sum() - energy_ratio_mean[channel]*(step) - energy_ratio[channel,0]   
            if step == 0 or reinit:
                U_temp.append(U_n)
                reinit = False
            else:
                u_n = min(U_temp)
                if U_n-u_n > lamb:
                    U_n , u_n = 0, 0
                    reinit = True
                    # Na_temp = step
                    if (len(U_temp) == 1) or (len(argrelextrema(array(U_temp), less)[0])==0):
                        Nd_temp = step-1
                    else:
                        Nd_temp = argrelextrema(array(U_temp), less)[0][-1] # When U is a 1D array, the output of argrelextrema is a tuple that contains an array in the first position. That array is the location of the local minima, the last one is Nd
                    break
                U_temp.append(U_n)
        Nd.append(Nd_temp)

    N0 = nanmin(Nd) # Reference time (first detection time)
    
    Ei = []
    for channel in range(energy_ratio.shape[0]):
        if isnan(Nd[channel]):
            Ei.append(0)
        else:
            Ei.append( (1/(Nd[channel] - N0 + 1))  * energy_ratio[channel,Nd[channel]:].sum() )
  
        

def PAC(signal1, signal2, fs, **kwars):
    """ Phase-amplitude coupling """   
    
    low = band_pass_filter(signal1, fs, "delta") #delta
    high = band_pass_filter(signal2, fs, "low_gamma") #low gamma
    # instantaneous phase angle 
    low_hil = hilbert(low)
    low_phase_angle = unwrap(angle(low_hil)) 
    # envelope extraction
    high_env_hil = hilbert(abs(hilbert(high))) 
    high_phase_angle = unwrap(angle(high_env_hil))
    # phases Difference
    phase_dif = low_phase_angle - high_phase_angle 
    # phase locking value
    plv = abs(mean(exp(complex(0,1)*phase_dif))) 
    return plv