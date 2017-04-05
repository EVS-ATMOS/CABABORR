#!/usr/bin/env python
"""
cababorr.process_cpol_bom
=========================
top level script for processing a cpol file from command line
on systems at the Australian Bureau of Meteorology.
CPOL Level 1b main production line.

@title: CPOL_PROD_1b
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 04/04/2017
@version: 0.5

.. autosummary::
    :toctree: generated/

    plot_figure_check
    production_line
    main
"""
# Python Standard Library
import os
import sys
import glob
import time
import logging
import argparse
import datetime
import warnings
from multiprocessing import Pool

# Other Libraries -- Matplotlib must be called first
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as pl

import pyart
import netCDF4
import numpy as np

# Custom modules.
import processing_code as radar_codes


def plot_figure_check(radar, gatefilter, outfilename):
    """
    Plot figure of old/new radar parameters for checking purpose.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        gatefilter:
            The Gate filter.
        outfilename:
            Name given to the output netcdf data file.
    """
    # Checking if figure already exists.
    outfile = os.path.basename(outfilename)
    outfile = outfile[:-2] + "png"
    outfile = os.path.join(FIGURE_CHECK_PATH, outfile)
    if os.path.isfile(outfile):
        logger.error('Figure already exists')
        return None

    # Initializing figure.
    gr = pyart.graph.RadarDisplay(radar)
    fig, the_ax = pl.subplots(6, 2, figsize=(12, 30), sharex=True, sharey=True)
    the_ax = the_ax.flatten()
    # Plotting reflectivity
    gr.plot_ppi('DBZ', ax = the_ax[0], vmin=-10, vmax=70)
    gr.plot_ppi('DBZ_CORR', ax = the_ax[1], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSRef, vmin=-10, vmax=70)

    gr.plot_ppi('ZDR', ax = the_ax[2], vmin=-5, vmax=10)  # ZDR
    gr.plot_ppi('ZDR_CORR', ax = the_ax[3], gatefilter=gatefilter, vmin=-5, vmax=10)

    gr.plot_ppi('PHIDP', ax = the_ax[4], vmin=0, vmax=180)
    gr.plot_ppi('PHIDP_CORR', ax = the_ax[5], gatefilter=gatefilter, vmin=0, vmax=180)

    gr.plot_ppi('VEL', ax = the_ax[6], cmap=pyart.graph.cm.NWSVel, vmin=-15, vmax=15)
    gr.plot_ppi('VEL_UNFOLDED', ax = the_ax[7], gatefilter=gatefilter, cmap=pyart.graph.cm.NWSVel, vmin=-15, vmax=15)

    gr.plot_ppi('SNR', ax = the_ax[8])
    gr.plot_ppi('KDP', ax = the_ax[9], gatefilter=gatefilter, vmin=-1, vmax=1)

    gr.plot_ppi('sounding_temperature', ax = the_ax[10], cmap='YlOrRd', vmin=-10, vmax=30)
    gr.plot_ppi('LWC', ax = the_ax[11], norm=colors.LogNorm(vmin=0.01, vmax=10), gatefilter=gatefilter, cmap='YlOrRd')

    for ax_sl in the_ax:
        gr.plot_range_rings([50, 100, 150], ax=ax_sl)
        ax_sl.axis((-150, 150, -150, 150))

    pl.tight_layout()
    pl.savefig(outfile)  # Saving figure.

    return None


def production_line(radar_file_name):
    """
    Production line for correcting and estimating CPOL data radar parameters.
    The naming convention for these parameters is assumed to be DBZ, ZDR, VEL,
    PHIDP, KDP, SNR, RHOHV, and NCP. KDP, NCP, and SNR are optional and can be
    recalculated.

    Parameters:
    ===========
        radar_file_name: str
            Name of the input radar file.
    """
    # Create output file name and check if it already exists.
    outfilename = os.path.basename(radar_file_name)
    outfilename = outfilename.replace("level1a", "level1b")
    outpath = os.path.expanduser('~')
    outfilename = os.path.join(outpath, outfilename)
    if os.path.isfile(outfilename):
        logger.error('Output file already exists. Nothing done.')
        print('Output file already exists. Nothing done.')
        return None

    # Read the input radar file.
    try:
        radar = pyart.io.read(radar_file_name)
        logger.info("Opening %s", radar_file_name)
    except:
        logger.error("MAJOR ERROR: Can't read input file named {}".format(radar_file_name))
        return None

    # Get the date and start the chrono.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    if radar_start_date.year > 2012:
        refold_velocity = True
    else:
        refold_velocity = False

    start_time = time.time()

    # Compute SNR
    height, temperature, snr = radar_codes.snr_and_sounding(radar, SOUND_DIR, 'DBZ')
    radar.add_field('sounding_temperature', temperature, replace_existing = True)
    radar.add_field('height', height, replace_existing = True)
    try:
        radar.fields['SNR']
        logger.info('SNR already exists.')
    except KeyError:
        radar.add_field('SNR', snr, replace_existing = True)
        logger.info('SNR saved.')

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing = True)
    logger.info('RHOHV corrected.')

    # Get filter
    gatefilter = radar_codes.do_gatefilter(radar)
    logger.info('Filter initialized.')

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

    # Estimate KDP
    try:
        radar.fields['KDP']
        logger.info('KDP already exists')
    except KeyError:
        logger.info('We need to estimate KDP')
        kdp_con = radar_codes.estimate_kdp(radar, gatefilter)
        radar.add_field('KDP', kdp_con, replace_existing=True)
        logger.info('KDP estimated.')

    # Bringi PHIDP/KDP
    phidp_bringi, kdp_bringi = radar_codes.bringi_phidp_kdp(radar)
    radar.add_field_like('PHIDP', 'PHIDP_BRINGI', phidp_bringi, replace_existing=True)
    radar.add_field_like('KDP', 'KDP_BRINGI', kdp_bringi, replace_existing=True)
    radar.fields['PHIDP_BRINGI']['long_name'] = "bringi_" + radar.fields['PHIDP_BRINGI']['long_name']
    radar.fields['KDP_BRINGI']['long_name'] = "bringi_" + radar.fields['KDP_BRINGI']['long_name']
    logger.info('KDP/PHIDP Bringi estimated.')

    # Unfold PHIDP, refold VELOCITY
    phidp_unfold, vdop_refolded = radar_codes.unfold_phidp_vdop(radar, unfold_vel=refold_velocity)
    radar.add_field_like('PHIDP', 'PHIDP_CORR', phidp_unfold, replace_existing=True)
    if vdop_refolded is not None:
        logger.info('Doppler velocity needs to be refolded.')
        radar.add_field_like('VEL', 'VEL_CORR', vdop_refolded, replace_existing=True)

    # Unfold VELOCITY
    try:
        radar.fields['VEL_CORR']
        vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter, vel_name='VEL_CORR')
    except KeyError:
        vdop_unfold = radar_codes.unfold_velocity(radar, gatefilter, vel_name='VEL')
    radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing = True)
    logger.info('Doppler velocity unfolded.')

    # Correct Attenuation ZH
    atten_spec, zh_corr = radar_codes.correct_attenuation_zh(radar)
    radar.add_field_like('DBZ', 'DBZ_CORR', zh_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zh', atten_spec, replace_existing=True)
    logger.info('Attenuation on reflectivity corrected.')

    # Correct Attenuation ZDR
    atten_spec_zdr, zdr_corr = radar_codes.correct_attenuation_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', zdr_corr, replace_existing=True)
    radar.add_field('specific_attenuation_zdr', atten_spec_zdr, replace_existing=True)
    logger.info('Attenuation on ZDR corrected.')

    # Hydrometeors classification
    hydro_class = radar_codes.hydrometeor_classification(radar)
    radar.add_field('HYDRO', hydro_class, replace_existing=True)
    logger.info('Hydrometeors classification estimated.')

    # Liquid/Ice Mass
    liquid_water_mass, ice_mass = radar_codes.liquid_ice_mass(radar)
    radar.add_field('LWC', liquid_water_mass)
    radar.add_field('IWC', ice_mass)
    logger.info('Liquid/Ice mass estimated.')

    # Treatment is finished!
    end_time = time.time()
    logger.info("Treatment for %s done in %f seconds.", os.path.basename(outfilename), (end_time - start_time))

    # Plot check figure.
    logger.info('Plotting figure')
    plot_figure_check(radar, gatefilter, outfilename)
    figure_time = time.time()
    logger.info("Figure for %s plotted in %f seconds.", os.path.basename(outfilename), (figure_time - end_time))

    # Rename fields and remove unnecessary ones.
    radar.add_field('DBZ', radar.fields.pop('DBZ_CORR'), replace_existing=True)
    radar.add_field('RHOHV', radar.fields.pop('RHOHV_CORR'), replace_existing=True)
    radar.add_field('ZDR', radar.fields.pop('ZDR_CORR'), replace_existing=True)
    radar.add_field('PHIDP', radar.fields.pop('PHIDP_CORR'), replace_existing=True)
    try:
        radar.fields['VEL_CORR']
        radar.add_field('VEL', radar.fields.pop('VEL_CORR'), replace_existing=True)
    except KeyError:
        pass
    radar.add_field('VEL_RAW', radar.fields.pop('VEL'), replace_existing=True)
    radar.add_field('VEL', radar.fields.pop('VEL_UNFOLDED'), replace_existing=True)

    # Hardcode mask
    for mykey in radar.fields:
        if (mykey == 'sounding_temperature' or mykey == 'height' or
            mykey == 'SNR' or mykey == 'NCP' or mykey == 'HYDRO'):
            continue
        else:
            radar.fields[mykey]['data'] = radar_codes.filter_hardcoding(radar.fields[mykey]['data'], gatefilter)
            logger.info('Hardcoding gatefilter for %s.', mykey)

    # Write results
    logger.info('Saving data')
    pyart.io.write_cfradial(outfilename, radar, format='NETCDF4')
    logger.info('Saving %s took %f seconds.', os.path.basename(outfilename), (time.time() - figure_time))

    return None


def main():
    flist = glob.glob("../data/*.nc")
    production_line(flist[0])

    return None


if __name__ == '__main__':
    """
    Global variables definition and logging file initialisation.
    """
    welcome_msg = "Leveling treatment of CPOL data from level 1a to level 1b."

    parser = argparse.ArgumentParser(description=welcome_msg)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')

    parser.add_argument(
        '-s',
        '--start-date',
        dest='start_date',
        default=None,
        type=str,
        help='Starting date.')

    parser.add_argument(
        '-e',
        '--end-date',
        dest='end_date',
        default=None,
        type=str,
        help='Ending date.')

    args = parser.parse_args()
    NCPU = args.ncpu
    START_DATE = args.start_date
    END_DATE = args.end_date

    INPATH = "/g/data2/rr5/vhl548/CPOL_level_1/"
    OUTPATH = "/g/data2/rr5/vhl548/CPOL_PROD_1b/"
    SOUND_DIR = "/data/vlouf/data/soudings_netcdf/"
    FIGURE_CHECK_PATH = os.path.expanduser('~')

    if not (START_DATE and END_DATE):
        parser.error("Starting and ending date required.")

    try:
        datetime.datetime.strptime(START_DATE, "%Y%m%d")
        datetime.datetime.strptime(END_DATE, "%Y%m%d")
    except:
        print("Did not understand the date format. Must be YYYYMMDD.")
        sys.exit()

    log_file_name =  os.path.join(os.path.expanduser('~'), 'cpol_level1b.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file_name,
        filemode='a+')
    logger = logging.getLogger(__name__)

    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
