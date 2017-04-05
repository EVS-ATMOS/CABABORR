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
# import skfuzzy as fuzz

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
