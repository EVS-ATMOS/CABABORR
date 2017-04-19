"""
Common processing code for CPOL.

.. autosummary::
    :toctree: generated/

    bringi_phidp_kdp
    compute_attenuation
    correct_attenuation_zdr
    correct_attenuation_zh
    correct_rhohv
    correct_zdr
    estimate_kdp
    hydrometeor_classification
    kdp_from_phidp_finitediff
    liquid_ice_mass
    nearest
    refold_vdop
    snr_and_sounding
    unfold_phi
    unfold_phidp_vdop
    unfold_velocity
"""
# Python Standard Library
import os
import glob
import time
import copy
import fnmatch
import datetime
from copy import deepcopy

# Other Libraries
import pyart
import netCDF4
import numpy as np
import skfuzzy as fuzz

from numba import jit, int32, float32

from scipy import ndimage, signal, integrate, interpolate
from scipy.stats import linregress
from scipy.ndimage.filters import convolve1d
from scipy.integrate import cumtrapz

from csu_radartools import csu_kdp, csu_liquid_ice_mass, csu_fhc


def bringi_phidp_kdp(radar, refl_name='DBZ', phidp_name='PHIDP'):
    """
    Compute PHIDP and KDP using Bringi's algorithm.

    Parameters:
    ===========
        radar:
            Py-ART radar structure
        refl_name: str
            Reflectivity field name
        phidp_name: str
            PHIDP field name

    Returns:
    ========
        fdN: array
            PhiDP Bringi
        kdN: array
            KDP Bringi
    """

    refl = radar.fields[refl_name]['data'].filled(fill_value = np.NaN)
    phidp = radar.fields[phidp_name]['data'].filled(fill_value = np.NaN)
    r = radar.range['data']

    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    dr = (r[1] - r[0])  # m

    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(dp=phidp, dz=refl, rng=rng2d/1000.0, thsd=6, gs=dr, window=3, bad=np.NaN)

    fdN = np.ma.masked_where(np.isnan(fdN), fdN)
    kdN = np.ma.masked_where(np.isnan(kdN), kdN)

    return fdN, kdN


def compute_attenuation(kdp, alpha = 0.08, dr = 0.25):
    """
    Alpha is defined by Ah=alpha*Kdp, beta is defined by Ah-Av=beta*Kdp.
    From Bringi et al. (2003)

    Parameters:
    ===========
        kdp: array
            Specific phase.
        alpha: float
            Parameter being defined by Ah = alpha*Kdp
        dr: float
            Gate range in km.

    Returns:
    ========
        atten_specific: array
            Specfific attenuation (dB/km)
        atten: array
            Cumulated attenuation (dB)
    """
    kdp[kdp < 0] = 0
    atten_specific = alpha*kdp
    atten_specific[np.isnan(atten_specific)] = 0
    atten = 2 * np.cumsum(atten_specific, axis=1) * dr

    return atten_specific, atten


def correct_attenuation_zdr(radar, zdr_name='ZDR', kdp_name='KDP'):
    """
    Correct attenuation on differential reflectivity.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        kdp_name: str
            KDP field name.

    Returns:
    ========
        atten_meta: dict
            Specific attenuation.
        zdr_corr: array
            Attenuation corrected differential reflectivity.
    """
    r = radar.range['data']
    zdr = radar.fields[zdr_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.016, dr=dr)
    zdr_corr = zdr + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zdr',
                  'long_name': 'Differential reflectivity specific attenuation'}

    return atten_meta, zdr_corr


def correct_attenuation_zh(radar, refl_name='DBZ', kdp_name='KDP'):
    """
    Correct attenuation on reflectivity.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        kdp_name: str
            KDP field name.

    Returns:
    ========
        atten_meta: dict
            Specific attenuation.
        zh_corr: array
            Attenuation corrected reflectivity.
    """
    r = radar.range['data']
    refl = radar.fields[refl_name]['data']
    kdp = radar.fields[kdp_name]['data']

    dr = (r[1] - r[0]) / 1000  # km

    atten_spec, atten = compute_attenuation(kdp, alpha=0.08, dr=dr)
    zh_corr = refl + atten

    atten_meta = {'data': atten_spec, 'units': 'dB/km', 'standard_name': 'specific_attenuation_zh',
                  'long_name': 'Reflectivity specific attenuation'}

    return atten_meta, zh_corr


def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]['data']
    snr = radar.fields[snr_name]['data']
    natural_snr = 10**(0.1*snr)
    rho_corr = rhohv / (1 + 1/natural_snr)

    return rho_corr


def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]['data']
    snr = radar.fields[snr_name]['data']
    alpha = 1.48
    natural_zdr = 10**(0.1*zdr)
    natural_snr = 10**(0.1*snr)
    corr_zdr = 10*np.log10((alpha*natural_snr*natural_zdr) / (alpha*natural_snr + alpha - natural_zdr))

    return corr_zdr


def do_gatefilter(radar, refl_name='DBZ', rhohv_name='RHOHV'):
    """
    Basic filtering

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_outside(refl_name, -20, 90)
    gf.exclude_below(rhohv_name, 0.5)

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld


def estimate_kdp(radar, gatefilter, phidp_name='PHIDP'):
    """
    Estimate KDP.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            Radar GateFilter (excluding bad data).
        phidp_name: str
            PHIDP field name.

    Returns:
    ========
        kdp_field: dict
            KDP.
    """
    phidp = radar.fields[phidp_name]['data'].data
    r = radar.range['data']

    phidp[gatefilter.gate_excluded] = np.NaN
    dr = (r[1] - r[0]) / 1000  # km

    kdp_data = kdp_from_phidp_finitediff(phidp, dr=dr)
    kdp_field = {'data': kdp_data, 'units': 'degrees/km', 'standard_name': 'specific_differential_phase_hv',
                 'long_name': 'Specific differential phase (KDP)'}

    return kdp_field


def filter_hardcoding(my_array, nuke_filter, bad=-9999):
    """
    Harcoding GateFilter into an array.

    Parameters:
    ===========
        my_array: array
            Array we want to clean out.
        nuke_filter: gatefilter
            Filter we want to apply to the data.
        bad: float
            Fill value.

    Returns:
    ========
        to_return: masked array
            Same as my_array but with all data corresponding to a gate filter
            excluded.
    """
    filt_array = np.ma.masked_where(nuke_filter.gate_excluded, my_array)
    filt_array.set_fill_value(bad)
    filt_array = filt_array.filled(fill_value=bad)
    to_return = np.ma.masked_where(filt_array == bad, filt_array)
    return to_return


def hydrometeor_classification(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                               kdp_name='KDP', rhohv_name='RHOHV_CORR',
                               temperature_name='sounding_temperature',
                               height_name='height'):
    """
    Compute hydrometeo classification.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        kdp_name: str
            KDP field name.
        rhohv_name: str
            RHOHV field name.
        temperature_name: str
            Sounding temperature field name.
        height: str
            Gate height field name.

    Returns:
    ========
        hydro_meta: dict
            Hydrometeor classification.
    """
    refl = radar.fields[refl_name]['data']
    zdr = radar.fields[zdr_name]['data']
    kdp = radar.fields[kdp_name]['data']
    rhohv = radar.fields[rhohv_name]['data']
    radar_T = radar.fields[temperature_name]['data']
    radar_z = radar.fields[height_name]['data']

    scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=True, band='C', T=radar_T)

    hydro = np.argmax(scores, axis=0) + 1
    fill_value = -32768
    hydro_data = np.ma.asanyarray(hydro)
    hydro_data.mask = hydro_data == fill_value

    the_comments = "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; " +\
                   "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"

    hydro_meta = {'data': hydro_data, 'units': ' ', 'long_name': 'Hydrometeor classification',
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments }

    return hydro_meta


def kdp_from_phidp_finitediff(phidp, L=7, dr=1.):
    """
    Retrieves KDP from PHIDP by applying a moving window range finite
    difference derivative. Function from wradlib.

    Parameters
    ----------
    phidp : multi-dimensional array
        Note that the range dimension must be the last dimension of
        the input array.
    L : integer
        Width of the window (as number of range gates)
    dr : gate length in km
    """

    assert (L % 2) == 1, \
        "Window size N for function kdp_from_phidp must be an odd number."
    # Make really sure L is an integer
    L = int(L)
    kdp = np.zeros(phidp.shape)
    for r in range(int(L / 2), phidp.shape[-1] - int(L / 2)):
        kdp[..., r] = (phidp[..., r + int(L / 2)] -
                       phidp[..., r - int(L / 2)]) / (L - 1)
    return kdp / 2. / dr


@jit(cache=True)
def liquid_ice_mass(radar, refl_name='DBZ_CORR', zdr_name='ZDR_CORR',
                    temperature_name='sounding_temperature', height_name='height'):
    """
    Compute the liquid/ice water content using the csu library.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        zdr_name: str
            ZDR field name.
        temperature_name: str
            Sounding temperature field name.
        height: str
            Gate height field name.

    Returns:
    ========
        liquid_water_mass: dict
            Liquid water content.
        ice_mass: dict
            Ice water content.
    """
    refl = radar.fields[refl_name]['data']
    zdr = radar.fields[zdr_name]['data']
    radar_T = radar.fields[temperature_name]['data']
    radar_z = radar.fields[height_name]['data']

    liquid_water_mass, ice_mass = csu_liquid_ice_mass.calc_liquid_ice_mass(refl, zdr, radar_z/1000, T=radar_T, method='cifelli')

    liquid_water_mass = {'data': liquid_water_mass, 'units': 'g m-3', 'long_name': \
                         'Liquid Water Content', 'standard_name': 'liquid_water_content'}
    ice_mass = {'data': ice_mass, 'units': 'g m-3', 'long_name': 'Ice Water Content',
                'standard_name': 'ice_water_content'}

    return liquid_water_mass, ice_mass


def nearest(items, pivot):
    """
    Find the nearest item.

    Parameters:
    ===========
        items:
            List of item.
        pivot:
            Item we're looking for.

    Returns:
    ========
        item:
            Value of the nearest item found.
    """
    return min(items, key=lambda x: abs(x - pivot))


def refold_vdop(vdop_art, v_nyq_vel, rth_position):
    """
    Refold Doppler velocity from PHIDP folding position.

    Parameters:
    ===========
        vdop_art: array
            Doppler velocity
        v_nyq_vel: float
            Nyquist velocity.
        rth_position: list
            Folding position of PHIDP along axis 2 (length of rth_position is
            length of axis 1).

    Returns:
    ========
        new_vdop: array
            Properly folded doppler velocity.
    """
    new_vdop = vdop_art
    for j in range(len(rth_position)):
        i = rth_position[j]
        if i == 0:
            continue
        else:
            new_vdop[j, i:] += v_nyq_vel

    pos = (vdop_art > v_nyq_vel)
    new_vdop[pos] = new_vdop[pos] - 2*v_nyq_vel

    return new_vdop


def snr_and_sounding(radar, soundings_dir=None, refl_field_name='DBZ'):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.

    Parameters:
    ===========
        radar:
        soundings_dir: str
            Path to the radiosoundings directory.
        refl_field_name: str
            Name of the reflectivity field.

    Returns:
    ========
        z_dict: dict
            Altitude in m, interpolated at each radar gates.
        temp_info_dict: dict
            Temperature in Celsius, interpolated at each radar gates.
        snr: dict
            Signal to noise ratio.
    """

    if soundings_dir is None:
        soundings_dir = "/g/data2/rr5/vhl548/soudings_netcdf/"

    # Getting radar date.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

    # Listing radiosounding files.
    sonde_pattern = datetime.datetime.strftime(radar_start_date, 'YPDN_%Y%m%d*')
    all_sonde_files = sorted(os.listdir(soundings_dir))

    try:
        # The radiosoundings for the exact date exists.
        sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]
    except IndexError:
        # The radiosoundings for the exact date does not exist, looking for the
        # closest date.
        dtime = [datetime.datetime.strptime(dt, 'YPDN_%Y%m%d_%H.nc') for dt in all_sonde_files]
        closest_date = nearest(dtime, radar_start_date)
        sonde_pattern = datetime.datetime.strftime(closest_date, 'YPDN_%Y%m%d*')
        radar_start_date = closest_date
        sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]

    interp_sonde = netCDF4.Dataset(os.path.join(soundings_dir, sonde_name))
    temperatures = interp_sonde.variables['temp'][:]
    times = interp_sonde.variables['time'][:]
    heights = interp_sonde.variables['height'][:]

    my_profile = pyart.retrieve.fetch_radar_time_profile(interp_sonde, radar)
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temperatures, my_profile['height'], radar)
    temp_info_dict = {'data': temp_dict['data'],
                 'long_name': 'Sounding temperature at gate',
                 'standard_name' : 'temperature',
                 'valid_min' : -100,
                 'valid_max' : 100,
                 'units' : 'degrees Celsius',
                 'comment': 'Radiosounding date: %s' % (radar_start_date.strftime("%Y/%m/%d"))}
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name)

    return z_dict, temp_info_dict, snr

def snr_and_sounding_interp_sonde(radar, soundings_dir):
    radar_start_date = netCDF4.num2date(radar.time['data'][0],
                                        radar.time['units'])
    sonde_pattern = datetime.datetime.strftime(radar_start_date,
                        '*interpolatedsondeC1.c1.%Y%m%d.*')
    all_sonde_files = os.listdir(soundings_dir)
    sonde_name = fnmatch.filter(all_sonde_files, sonde_pattern)[0]
    print(sonde_pattern,sonde_name)
    interp_sonde = netCDF4.Dataset(os.path.join( soundings_dir, sonde_name))
    temperatures = interp_sonde.variables['temp'][:]
    times = interp_sonde.variables['time'][:]
    heights = interp_sonde.variables['height'][:]
    my_profile = pyart.retrieve.fetch_radar_time_profile(interp_sonde, radar)
    info_dict = {'long_name': 'Sounding temperature at gate',
                 'standard_name' : 'temperature',
                 'valid_min' : -100,
                 'valid_max' : 100,
                 'units' : 'degrees Celsius'}
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(my_profile['temp'],
                                             my_profile['height']*1000.0,
                                             radar)
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar)
    return z_dict, temp_dict, snr

def get_texture(radar):
    #Some bodging because CPOL masks higher gates
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    foo = deepcopy(radar.fields['velocity']['data'].data)
    mask = radar.fields['velocity']['data'].mask
    rand = (np.random.rand(foo.shape[0], foo.shape[1]) - .5) * radar.instrument_parameters['nyquist_velocity']['data'][0]*2.
    foo[mask] = rand[mask]
    start_time = time.time()
    data = ndimage.filters.generic_filter(foo,
                                                pyart.util.interval_std,
                                                size = (4,4),
                                               extra_arguments = (-nyq, nyq))
    total_time = time.time() - start_time
    print(total_time)
    filtered_data = ndimage.filters.median_filter(data, size = (4,4))
    texture_field = pyart.config.get_metadata('velocity')
    texture_field['data'] = filtered_data
    return texture_field


def unfold_phi(phidp, kdp):
    """
    Alternative phase unfolding which completely relies on Kdp.
    This unfolding should be used in oder to iteratively reconstruct
    phidp and Kdp. Function from wradlib.

    Parameters:
    ===========
        phidp : array
        kdp : array

    Returns:
    ========
        phidp: array
        rth_pos: list
            Position of folding along axis 1.
    """
    # unfold phidp
    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))
    kdp = kdp.reshape((-1, shape[-1]))

    rth_pos = np.zeros((phidp.shape[0]), dtype=np.int32)

    for beam in range(phidp.shape[0]):
        below_th3 = kdp[beam] < -20
        try:
            idx1 = np.where(below_th3)[0][2]
            phidp[beam, idx1:] += 360
            rth_pos[beam] = idx1
        except Exception:
            pass

    if len(rth_pos[rth_pos != 0]) == 0:
        rth_pos = None

    return phidp.reshape(shape), rth_pos


def unfold_phidp_vdop(radar, phidp_name='PHIDP', kdp_name='KDP', vel_name='VEL', unfold_vel=False):
    """
    Unfold PHIDP and refold Doppler velocity.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        kdp_name:
            KDP field name.
        phidp_name: str
            PHIDP field name.
        vel_name: str
            VEL field name.
        unfold_vel: bool
            Switch the Doppler velocity refolding

    Returns:
    ========
        phidp_unfold: dict
            Unfolded PHIDP.
        vdop_refolded: dict
            Refolded Doppler velocity.
    """
    fdN = radar.fields[phidp_name]['data']
    kdN = radar.fields[kdp_name]['data']
    vdop_art = radar.fields[vel_name]['data']

    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        v_nyq_vel = np.max(np.abs(vdop_art))

    phidp_unfold, pos_unfold = unfold_phi(fdN, kdN)

    vdop_refolded = None
    if unfold_vel:
        if pos_unfold is not None:
            vdop_refolded = refold_vdop(vdop_art, v_nyq_vel, pos_unfold)

    return phidp_unfold, vdop_refolded


def unfold_velocity(radar, my_gatefilter, vel_name=None):
    """
    Unfold Doppler velocity using Py-ART region based algorithm.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        my_gatefilter:
            GateFilter
        vel_name: str
            Name of the Doppler velocity field.

    Returns:
    ========
        vdop_vel: dict
            Unfolded Doppler velocity.
    """
    if vel_name is None:
        raise ValueError('Name of Doppler velocity field not provided.')

    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except:
        vdop_art = radar.fields[vel_name]['data']
        v_nyq_vel = np.max(np.abs(vdop_art))

    vdop_vel = pyart.correct.dealias_region_based(radar, vel_field=vel_name, gatefilter=my_gatefilter, nyquist_vel=v_nyq_vel)
    # vdop_vel['standard_name'] = radar.fields[vel_name]['standard_name']

    return vdop_vel

#moment : [[start_up, finish_up, start_down, finish_down], weight]

#moment : [[start_up, finish_up, start_down, finish_down], weight]
def cum_score_fuzzy_logic(radar, mbfs = None,
                          debug = False, ret_scores = False,
                          hard_const = None):
    if mbfs == None:
        second_trip = {'velocity_texture' : [[0,0,1.8,2], 1.0],
                       'cross_correlation_ratio' : [[.5,.7,1,1], 0.0],
                       'normalized_coherent_power' : [[0,0,.5,.6], 3.0],
                       'height': [[0,0,5000,8000], 1.0],
                       'sounding_temperature' : [[-100,-100,100,100], 0.0],
                       'SNR' : [[15,20, 1000,1000],1.0]}

        rain = {'differential_phase_texture' : [[0,0,80,90], 1.0],
                       'cross_correlation_ratio' : [[0.94,0.96,1,1], 1.0],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 1.0],
                       'height': [[0,0,5000,6000], 0.0],
                       'sounding_temperature' : [[0,3,100,100], 2.0],
                       'SNR' : [[8,10, 1000,1000], 1.0]}

        snow = {'differential_phase_texture' : [[0,0,80,90], 1.0],
                       'cross_correlation_ratio' : [[0.85,0.9,1,1], 1.0],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 1.0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-100,-100,0,1.], 2.0],
                       'SNR' : [[8,10, 1000,1000], 1.0]}

        no_scatter = {'differential_phase_texture' : [[90,90,400,400], 0.0],
                       'cross_correlation_ratio' : [[0,0,0.1,0.2], 0.0],
                       'normalized_coherent_power' : [[0,0,0.1,0.2], 0.0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-100,-100,100,100], 0.0],
                       'SNR' : [[-100,-100, 8,10], 6.0]}

        melting = {'differential_phase_texture' : [[20,30,80,90], 0.0],
                       'cross_correlation_ratio' : [[0.6,0.7,.94,.96], 4.],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-1.,0,3.5,5], 2.],
                       'SNR' : [[8,10, 1000,1000], 0.0]}

        mbfs = {'multi_trip': second_trip, 'rain' : rain,
                'snow' :snow, 'no_scatter' : no_scatter, 'melting' : melting}
    flds = radar.fields
    scores = {}
    for key in mbfs.keys():
        if debug: print('Doing ' + key)
        this_score = np.zeros(\
                flds[list(flds.keys())[0]]['data'].shape).flatten() * 0.0
        for MBF in mbfs[key].keys():
            this_score = fuzz.trapmf(flds[MBF]['data'].flatten(),
                         mbfs[key][MBF][0] )*mbfs[key][MBF][1] + this_score

        this_score = this_score.reshape(\
                flds[list(flds.keys())[0]]['data'].shape)
        scores.update({key: ndimage.filters.median_filter(\
                this_score, size = [3,4])})
    if hard_const != None:
        # hard_const = [[class, field, (v1, v2)], ...]
        for this_const in hard_const:
            if debug: print('Doing hard constraining ', this_const[0])
            key = this_const[0]
            const = this_const[1]
            fld_data = radar.fields[const]['data']
            lower = this_const[2][0]
            upper = this_const[2][1]
            const_area = np.where(np.logical_and(fld_data >= lower,
                fld_data <= upper))
            if debug: print(const_area)
            scores[key][const_area] = 0.0
    stacked_scores = np.dstack([scores[key] for key in scores.keys() ])
    #sum_of_scores = stacked_scores.sum(axis = 2)
    #print(sum_of_scores.shape)
    #norm_stacked_scores = stacked_scores
    max_score = stacked_scores.argmax(axis = 2)

    gid = {}
    gid['data'] = max_score
    gid['units'] = ''
    gid['standard_name'] = 'gate_id'

    strgs = ''
    i=0
    for key in scores.keys():
        strgs = strgs + str(i) + ':' + key + ','
        i = i + 1

    gid['long_name'] = 'Classification of dominant scatterer'
    gid['notes'] = strgs[0:-1]
    gid['valid_max'] = max_score.max()
    gid['valid_min'] = 0.0
    if ret_scores == False:
        rv = (gid, scores.keys())
    else:
        rv = (gid, scores.keys(), scores)
    return rv

def fix_rain_above_bb(gid_fld, rain_class, melt_class, snow_class):
    print(snow_class)
    new_gid = copy.deepcopy(gid_fld)
    for ray_num in range(new_gid['data'].shape[0]):
        if melt_class in new_gid['data'][ray_num, :]:
            max_loc = np.where(new_gid['data'][ray_num, :] == melt_class)[0].max()
            rain_above_locs = np.where(new_gid['data'][ray_num, max_loc:] == rain_class)[0] + max_loc
            new_gid['data'][ray_num, rain_above_locs] = snow_class
    return new_gid

def do_my_fuzz(radar):
    print('doing')
    second_trip = {'velocity_texture' : [[4.9,5.3,130.,130.], 5.0],
                   'cross_correlation_ratio' : [[.5,.7,1,1], 0.0],
                   'normalized_coherent_power' : [[0,0,.5,.6], 0.0],
                   'height': [[0,0,5000,8000], 0.0],
                   'sounding_temperature' : [[-100,-100,100,100], 0.0],
                   'SNR' : [[5,10, 1000,1000],1.0]}

    rain = {'velocity_texture' : [[0,0,4.9,5.3], 2.0],
                   'cross_correlation_ratio' : [[0.97,0.98,1,1], 1.0],
                   'normalized_coherent_power' : [[0.4,0.5,1,1], 0.0],
                   'height': [[0,0,5000,6000], 0.0],
                   'sounding_temperature' : [[2.,5.,100,100], 2.0],
                   'SNR' : [[8,10, 1000,1000], 1.0]}

    snow = {'velocity_texture' : [[0,0,4.9,5.3], 2.0],
                   'cross_correlation_ratio' : [[0.65,0.9,1,1], 1.0],
                   'normalized_coherent_power' : [[0.4,0.5,1,1], 0.0],
                   'height': [[0,0,25000,25000], 0.0],
                   'sounding_temperature' : [[-100,-100,.5,4.], 2.0],
                   'SNR' : [[8,10, 1000,1000], 1.0]}

    no_scatter = {'velocity_texture' : [[4.9,5.3,330.,330.], 2.0],
                   'cross_correlation_ratio' : [[0,0,0.1,0.2], 0.0],
                   'normalized_coherent_power' : [[0,0,0.1,0.2], 0.0],
                   'height': [[0,0,25000,25000], 0.0],
                   'sounding_temperature' : [[-100,-100,100,100], 0.0],
                   'SNR' : [[-100,-100, 5,10], 4.0]}

    melting = {'velocity_texture' : [[0,0,4.9,5.3], 0.0],
                   'cross_correlation_ratio' : [[0.6,0.65,.9,.96], 2.0],
                   'normalized_coherent_power' : [[0.4,0.5,1,1], 0],
                   'height': [[0,0,25000,25000], 0.0],
                   'sounding_temperature' : [[0,0.1,2,4], 4.0],
                   'SNR' : [[8,10, 1000,1000], 0.0]}

    mbfs = {'multi_trip': second_trip, 'rain' : rain,
            'snow' :snow, 'no_scatter' : no_scatter, 'melting' : melting}

    hard_const = [['melting' , 'sounding_temperature', (10, 100)],
                  ['multi_trip', 'height', (10000, 1000000)],
                  ['melting' , 'sounding_temperature', (-10000, -2)],
                  ['rain', 'sounding_temperature',(-1000,-5)],
                  ['melting', 'velocity_texture', (3,300)]]

    gid_fld, cats = cum_score_fuzzy_logic(radar,
            mbfs = mbfs, debug = True,
            hard_const = hard_const)
    rain_val = list(cats).index('rain')
    snow_val = list(cats).index('snow')
    melt_val = list(cats).index('melting')

    return fix_rain_above_bb(gid_fld, rain_val, melt_val, snow_val), cats

def extract_unmasked_data(radar, field, bad=-32768):
    """Simplify getting unmasked radar fields from Py-ART"""
    return radar.fields[field]['data'].filled(fill_value=bad)

def csu_to_field(field, radar, units='unitless',
                              long_name='Hydrometeor ID',
                              standard_name='Hydrometeor ID',
                              dz_field='ZC'):
    """
    Adds a newly created field to the Py-ART
    radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask',
                np.logical_or(masked_field.mask,
                    radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    return field_dict

def return_csu_kdp(radar):
    dzN = extract_unmasked_data(radar, 'reflectivity')
    dpN = extract_unmasked_data(radar, 'differential_phase')
    # Range needs to be supplied
    #as a variable, and it needs to be
    #the same shape as dzN, etc.
    rng2d, az2d = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    bt = time.time()
    kdN, fdN, sdN = csu_kdp.calc_kdp_bringi(
        dp=dpN, dz=dzN, rng=rng2d/1000.0, thsd=12, gs=250.0, window=5)
    print(time.time()-bt, 'seconds to run')
    csu_kdp_field = csu_to_field(kdN,
            radar, units='deg/km',
            long_name='Specific Differential Phase',
            standard_name='Specific Differential Phase',
            dz_field='reflectivity')
    csu_filt_dp = csu_to_field(fdN,
            radar, units='deg',
            long_name='Filtered Differential Phase',
            standard_name='Filtered Differential Phase',
            dz_field='reflectivity')
    csu_kdp_sd = csu_to_field(sdN, radar,
            units='deg',
            long_name='Standard Deviation of Differential Phase',
            standard_name='Standard Deviation of Differential Phase',
            dz_field='reflectivity')
    return  csu_kdp_field, csu_filt_dp, csu_kdp_sd

def retrieve_qvp(radar, hts, flds = None):
    if flds == None:
        flds = ['differential_phase',
            'cross_correlation_ratio',
            'spectrum_width',
            'reflectivity', 'differential_reflectivity']
    desired_angle = 20.0
    index = abs(radar.fixed_angle['data'] - desired_angle).argmin()
    ss = radar.sweep_start_ray_index['data'][index]
    se = radar.sweep_end_ray_index['data'][index]
    mid = int((ss + se)/2)
    z = radar.gate_altitude['data'][mid, :]
    qvp = {}
    for fld in flds:
        this_fld = radar.get_field(index, fld)[:, :].mean(axis = 0)
        intery = interpolate.interp1d(z,
                this_fld, bounds_error=False, fill_value='extrapolate')
        ithis = intery(hts)
        qvp.update({fld:ithis})
    qvp.update({'time': radar.time})
    qvp.update({'height': hts})
    return qvp


