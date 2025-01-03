import logging
import os
import pickle
from datetime import datetime
from glob import glob
import pandas as pd
from shutil import copy2, move

import numpy as np
from netCDF4 import Dataset
from scipy.linalg import svd
import scipy.signal

import matplotlib.pyplot as plt

logger = logging.getLogger("irs_logger")


def pca_expand(coef, PC):

    (nX, ntr) = coef.shape
    nvals = len(PC['m'])

    x = np.dot(coef, PC['u'][:, 0:ntr].T) + np.full((nX, 1), 1.) @ np.array([PC['m']])

    return x


def pca_ind(evals, nsamp, doplot=False):
    """
    Abstract:
        This script takes as input an array of eigenvalues and computes from them
        the IND, IE, RE, PCV, and other parameters.  But its main function is to compute
        the number of eigenvectors that should be used in the reconstruction of the PCA
        data, based upon the Error Indicator Function (IND).

    Reference:
        Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using
        Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238,
        https://doi.org/10.1175/JTECH1906.1.

    Author:
        Dave Turner
        SSEC / University of Wisconsin - Madison

        Ported to Python by Tyler Bell (CIWRO/NSSL) - Sept 2020


    :param evals: Array of eigenvalues
    :param nsamp: Number of samples
    :return:
    :rtype:
    """

    c = len(evals)

    f_ie = np.zeros(c-1)
    f_re = np.zeros(c-1)
    f_xe = np.zeros(c-1)
    f_ind = np.zeros(c-1)
    f_pcv = np.zeros(c-1)
    for n in range(c-1):
        f_ie[n] = np.sqrt( (n*np.sum(evals[n+1:c])) / (nsamp*c*(c-n)) )
        f_re[n] = np.sqrt( np.sum(evals[n+1:c]) / (nsamp*(c-n)) )
        f_xe[n] = np.sqrt( np.sum(evals[n+1:c]) / (nsamp * c) )
        f_pcv[n] = 100. * np.sum(evals[0:n+1]) / np.sum(evals)
        f_ind[n] = f_re[n] / (c-n)**2.

    idx = np.arange(0, c-1)

    if doplot:
        import matplotlib.pyplot as plt
        bar = np.where(idx >= 5)
        foo = np.argmin(f_ie[bar])
        print(f"The IE optimal number of PCs to use is {idx[bar][foo]}")
        foo = np.argmin(f_ind)
        print(f"The IND optimal number of PCs to use is {idx[foo]}")

        plt.subplot(2, 2, 1)
        plt.semilogx(idx, f_re)
        plt.title("Real Error")
        plt.subplot(2, 2, 2)
        plt.semilogx(idx, f_ie)
        plt.title("Imbedded Error")
        plt.subplot(2, 2, 3)
        plt.loglog(idx, f_ind)
        plt.title("Error Indicator Function")
        plt.show()


    foo = np.argmin(f_ind)
    xe = f_xe[foo]
    re = f_re[foo]
    pcv = f_pcv[foo]

    return idx[foo]


def pca_project(x, ntr, pc):
    """
    Abstract:
        Routine to project (compress) along eigenvectors determined from the
        principal component analysis.

    Original Author:
        Paolo Antonelli and Raymond Garcia
            University of Wisconsin - Madison

    Ported into IDL by:
        Dave Turner
            University of Wisconsin - Madison
            Pacific Northwest National Laboratory

    IDL version ported into Python by:
        Tyler Bell
            OU CIMMS/NSSL

    :param x: A two-dimensional matrix, with time as the second dimension
    :param ntr: Numberprincipal components to use in the projection
    :param pc: Dictionary containing PCA reconstruction matrix
    :return: Columns of principal component coefficients
    """

    (nX, nvals) = x.shape

    coef = x - np.full((1, nX), 1.).T @ np.array([pc['m']])
    coef = np.dot(pc['u'][:, 0:ntr].T, coef.T).T

    return coef

def find_spikes(signal,nf):
    """
    This routine finds spikes in any timeseries using a butterworth filter
    Author: Bianca Adler, CIRES/NOAA PSL
    Original Author: Matlab code written by Vincent Michaud-Belleau, LR Tech, 2024

    signal: 1-D time series of variable
    nf: parameter used for order of filter and critical frequency
    """

    sos= scipy.signal.butter(nf/2, 1/nf,output='sos')
    signal_filt = scipy.signal.sosfiltfilt(sos,signal);
    signal_diff = signal-signal_filt;

    signal_std = np.emath.sqrt(scipy.signal.sosfiltfilt(sos,signal_diff**2))
    signal_cir = (np.roll(signal_std,nf)+np.roll(signal_std,-nf))/2;

    ratio = abs(signal_diff)/signal_cir

    return ratio


def read_ch_radiance_data(files,sky_view_angle):
    """
    This routines reads in the radiance data from ch1 
    and tries to performs a quality control

    Author: Bianca Adler, CIRES/NOAA PSL
    
    input
    files: list of filenames to be read
    sky_view_angle: angle in degree of zenith
    
    output:
    secs: array of seconds 
    rad: matrix of radiances
    qcflag: array of quality flag
    wnum: array of wavenumbers
    """

    # Read in the radiance data
    for i, fn in enumerate(sorted(files)):
        logger.debug(f'Loading {fn}')

        nc = Dataset(fn)

        if 'wnum' in nc.variables.keys():
            wnum = nc['wnum'][:]
        elif 'wnum1' in nc.variables.keys():
            wnum = nc['wnum1'][:]
        else:
            logger.error(" Unable to find either wavenumber field in the data - aborting")
            return

        rad = nc['mean_rad'][:]
        foo = np.where(wnum < 3000)
        wnum = wnum[foo]
        rad = np.squeeze(rad[:, foo])
        bt = nc['base_time'][0]
        mirror = nc['sceneMirrorAngle'][:]
        resp = np.squeeze(nc['mean_resp'][:][:,foo])
        imag = np.squeeze(nc['mean_imag_rad'][:][:,foo])
        hatchopen = nc['hatchOpen'][:]

        try:
            to = nc['time_offset'][:]
        except IndexError:  # ASSISTs are different...
            bt *= 1e-3
            to = nc['time'][:]
        try:
            cqcflag = nc['missingDataFlag'][:]
        except IndexError:  # ASSIST doesn't have missingDataFlag
            cqcflag = np.zeros_like(to)
        
        try:
            logger.debug(f'Quality control ch data')
            #QC flag for ASSIST (for ch1 data)
            #1. Check for to =0 and remove
            #2. It checks for stuck mirror or periods contaminated by rain/snow (visible in very noisy radiances or drop in responsivity) and flags these periods
            #3. it inspects rad at 675 cm-1 for spikes
            #4. it inspects rad at 675 cm-1 for spikes
            
            #1.
            logger.debug(f'checking for to=0')
            idx=np.where(to>0)[0]
            logger.debug(f' {str(len(to)-len(idx))} samples found')
            rad=rad[idx,:]
            to=to[idx]
            mirror=mirror[idx]
            resp=resp[idx,:]
            imag=imag[idx,:]
            hatchopen=hatchopen[idx]

            #2. 
            imag=np.array(imag)
            resp=np.array(resp)
            # use wavenumber region 1580-1610
            wnidx=np.where((wnum>=1580) & (wnum<=1610))[0]
            imag=np.nanmean(imag[:,wnidx],axis=1)
            df=pd.DataFrame(imag)
            dfstd=df.rolling(50).std().to_numpy().squeeze()

            resp=np.nanmean(resp[:,wnidx],axis=1)
            idx=np.where((np.abs(dfstd)>1) | (np.abs(resp-np.nanmedian(resp))>10000))
            logger.debug(f' {str(len(idx[0]))} samples with stuck mirror found')
            cqcflag[idx]=1
            
            #3.
            dhatch=np.diff(np.concatenate((np.atleast_1d(hatchopen[0]),hatchopen)))
            #find all indices where hatch has moved
            idx=np.where(np.abs(dhatch)>1e-5)[0]
            #also include indices before and after hatch movement was detected
            #idxall=np.unique(np.sort(np.asarray([idx-1,idx,idx+1]).flatten()))
            idxall=np.unique(np.sort(np.asarray([idx-2,idx-1,idx,idx+1]).flatten()))
            #make sure that no indices are out of range
            idx=np.where((idxall>=0) & (idxall<len(cqcflag)))[0]
            idxall=idxall[idx]
            cqcflag[idxall]=1
            logger.debug(f' {str(len(idxall))} samples with hatch moved found')
            #also flag times when hatch is not open or undefined
            cqcflag[hatchopen<1]=1

            #4. 
            #use previous qcflag as well as sceneMirrorAngle if required
            if sky_view_angle is not None:
                if((sky_view_angle < 3) | (sky_view_angle > 357)):
                    mflag = (mirror < 3) | (mirror > 357)
                else:
                    mflag = (sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3)
            idx=np.where((mflag) & (cqcflag==0))[0]
            if len(idx)>0:
                wnidx=np.argmin(np.abs(wnum-675))
                rad_=rad[idx,wnidx]
                spikes_675=find_spikes(rad_,6)
                logger.debug(f' {str(len(np.where(spikes_675>5)[0]))} samples with spikes found')
                cqcflag[idx[spikes_675>5]]=1

        except:
            print('searching for stuck mirror, hatch position, and spikes  using ch1 data failed')

        nc.close()

        #to avoid precision error
        bt=bt.astype(np.float64)
        to=to.astype(np.float64)

        if sky_view_angle is not None:
            if((sky_view_angle < 3) | (sky_view_angle > 357)):
                bar = np.where((mirror < 3) | (mirror > 357))[0]
            else:
                bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

            to = to[bar]
            rad = rad[bar]
            cqcflag = cqcflag[bar]

        if i == 0:
            xsecs = bt+to
            xrad = rad.T
            xqcflag = cqcflag
        else:
            xsecs = np.append(xsecs, bt+to)
            xrad = np.append(xrad, rad.T, axis=1)
            xqcflag = np.append(xqcflag, cqcflag)

    secs = xsecs
    rad = np.squeeze(np.transpose(xrad))
    qcflag = xqcflag

    return secs, rad, qcflag, wnum

def read_sum_noise_data(secs,sdir,sfields,sky_view_angle):
    """
    This routines reads in the noise data from the summary files 
    and tries to performs a quality control for the ASSIST. 
    It interpolates the noise for wavenumber 675 cm-1

    Author: Bianca Adler, CIRES/NOAA PSL
    
    input
    secs: array of seconds from radiance data
    sdir: directory path of summary files
    sfields: array with wavenumber and noise variables names
    sky_view_angle: angle in degree of zenith
    
    output:
    nsecs: array of seconds 
    nrad: matrix of noise
    nqcflag: array of quality flag
    nwnum: array of wavenumbers
    """


    times = np.array([datetime.utcfromtimestamp(d) for d in secs])
    yyyymmdd = [d.strftime("%Y%m%d") for d in times]

    for ymd,i_ymd in zip(np.unique(yyyymmdd),range(len(np.unique(yyyymmdd)))):
        sfiles = glob(os.path.join(sdir, f'*sum*{ymd}*.cdf'))
        for j, sfile in enumerate(sorted(sfiles)):
            logger.debug(f'Loading {sfile}')
            nc = Dataset(sfile)
            nwnum = nc[sfields[0]][:]
            nrad = nc[sfields[1]][:]
            mirror = nc['sceneMirrorAngle'][:]
            bt = nc['base_time'][0]

            try:
                to = nc['time_offset'][:]
            except IndexError:  # ASSISTs are different...
                bt *= 1e-3
                to = nc['time'][:]

            #QC flag for ASSIST (for summary data)
            #1. It flags spikes caused by slow mirror movement (it may affect scene #1 since a HBB view is included in the average (based on suggestions from Vincent incent Michaud-Belleau, LR Tech
            try:
                cqcflag = np.zeros_like(to)
                #1.
                ratio_spikes_700=find_spikes(nc['mean_Tb_700_705'][:],6)
                ratio_spikes_985=find_spikes(nc['mean_Tb_985_990'][:],6)
                ratio_spikes_2295=find_spikes(nc['mean_Tb_2295_2300'][:],6)
                ratio_spikes_2510=find_spikes(nc['mean_Tb_2510_2515'][:],6)
                ratio_spikes_pch1=find_spikes(nc['p_pCh1'][:],6)
                ratio_spikes_pch2=find_spikes(nc['p_pCh2'][:],6)

                ratio_spikes_opaque = ratio_spikes_700+ratio_spikes_2295
                ratio_spikes_transp= ratio_spikes_985+ratio_spikes_2510
                ratio_spikes_ADC= ratio_spikes_pch1+ratio_spikes_pch2

                cond_spikes_opaque = ratio_spikes_opaque>24
                cond_spikes_transp= ratio_spikes_transp>24
                cond_spikes_ADC= ratio_spikes_ADC>32

                #make sure that spikes are detected at scene #1
                tdiff=np.median(np.diff(np.array(to)))
                cond_scene1=np.concatenate((np.atleast_1d(False),np.abs(np.diff((np.array(to))))>tdiff*2))

                #identify hatch and instrument conditions condiitions
                hatch=nc['hatchOpen'][:]
                hbbtemp=nc['calibrationHBBtemp'][:]
                hatch[hatch==-1]=-2
                # b=np.ones((2,1))/2
                b=1/2
                dhatch=np.diff(np.concatenate((np.atleast_1d(0),hatch)))
                #find all indices where hatch has moved
                idx=np.where(np.abs(dhatch)>1e-5)[0]
                #also include indices before and after hatch movement was detected
                # idxall=np.unique(np.sort(np.asarray([idx-1,idx,idx+1]).flatten()))
                idxall=np.unique(np.sort(np.asarray([idx-2,idx-1,idx,idx+1]).flatten()))
                idxall=idxall[(idxall>=0) & (idxall<len(hatch)) ]
                hatchStable_cond=np.ones(hatch.shape)
                hatchStable_cond[idxall]=0
                #convert to mask
                hatchStable_cond = hatchStable_cond > 0

                instrumentStable_cond=np.abs(hbbtemp-60)<0.1

                valid_cond =((hatchStable_cond & instrumentStable_cond) & cond_scene1)
                valid_cond[0:6]=True
                valid_cond[-6:]=True

                cond=(cond_spikes_opaque | cond_spikes_transp | cond_spikes_ADC) & valid_cond
                logger.debug(f' {str(len(np.where(cond)[0]))} samples in sum flagged')
                cqcflag[cond]=1

            except:
                print('searching for spikes (summary files')
                cqcflag = np.zeros_like(to)


            nc.close()

            #to avoid precision error
            bt=bt.astype(np.float64)
            to=to.astype(np.float64)

            if sky_view_angle is not None:
                if((sky_view_angle < 3) | (sky_view_angle > 357)):
                    bar = np.where((mirror < 3) | (mirror > 357))[0]
                else:
                    bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

                to = to[bar]
                nrad = nrad[bar]


            if i_ymd == 0:
                xsecs = bt + to
                xnrad = nrad.T
                xqcflag = cqcflag
            else:
                xsecs = np.append(xsecs, bt + to)
                xnrad = np.append(xnrad, nrad.T,axis=1)
                xqcflag = np.append(xqcflag, cqcflag)

    nsecs = xsecs
    nrad = np.transpose(xnrad)
    nqcflag = xqcflag

    #The noise spectra shows an artifical spike at around 675 cm-1, interpolate
    foo = np.argmin(np.abs(nwnum-675))
    nrad[:,foo]=np.nan
    #now interpolate over nan gap
    nans, x= np.isnan(nrad[:,:]), lambda z: z.nonzero()[0]
    nrad[nans]=np.interp(x(nans),x(~nans),nrad[~nans])

    return nsecs,nrad,nqcflag,nwnum

def create_irs_noise_filter(files, sdir, sfields,tdir, pcs_filename=None, 
                            use_median_noise=False, sky_view_angle=None):
    """
    Abstract:
        This function is designed to create the PCA the noise filter from IRS obvservations.  
        It uses a Principal Component Analysis technique published in Turner et al.
        JTECH 2006.  It is designed to use either ARM-formatted AERI, dmv-ncdf
        formatted AERI, or ASSIST data. 

    Author: Tyler Bell (CIWRO/NSSL, 2023) 2023
        Based on code from Dave Turner
        
    Reference:
        Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using
        Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238,
        https://doi.org/10.1175/JTECH1906.1.

    """

    # Read in the radiance data
    secs,rad,qcflag,wnum = read_ch_radiance_data(files,sky_view_angle)

    # Now read in the summary files
    nsecs,nrad,nqcflag,nwnum=read_sum_noise_data(secs,sdir,sfields,sky_view_angle)


    #Add ncqcflag to qcflag
    logger.debug(f"Merge QC flags for sum and Ch1")
    cqcflag=np.zeros(qcflag.shape)
    cqcflag=np.full((qcflag.shape),9)
    for i in range(len(secs)):
        dels=np.abs(secs[i]-nsecs)
        #sum and ch1 samples need to be less than 5 secs apart
        foo=np.where((dels  == np.min(dels)) & (np.min(dels) < 5))[0]
        if len(foo) == 1:
            cqcflag[i]=nqcflag[foo] or qcflag[i]
    qcflag=cqcflag

    # Quick check for consistency
    logger.debug(f"Adding median noise value to missing summary samples and limit summary samples to time stamps of rad data")
    medianNoise=np.mean(nrad,axis=0)
    zrad=np.full((rad.shape[0],len(nwnum)),np.nan)
    zsecs=np.full((rad.shape[0]),np.nan)
    for i in range(len(secs)):
        dels=np.abs(secs[i]-nsecs)
        foo=np.where((dels  == np.min(dels)) & (np.min(dels) < 5))[0]
        if len(foo) == 1:
            #print('use actual noise value')
            zrad[i,:]=nrad[foo,:]
            zsecs[i]=nsecs[foo]
        else:
            print('use median noise')
            zrad[i,:]=medianNoise
            zsecs[i]=secs[i]
    
    nrad=zrad
    nsecs=zsecs

    osecs = secs
    orad = rad.transpose()
    onrad = nrad.transpose()
    oqc = qcflag

    logger.info(f"Number of samples: {len(secs)}; Number of spectral channels: {len(wnum)}")

    # Select only the good data (simple check)
    if min(wnum) < 600:
        logger.debug("I believe I'm working with ch1 data")
        foo = np.where(wnum >= 900)
        # We assume 30 deg C, which results in an upper threshold of 122RU, which is reasonable for many locations.
        minv = -2
        maxv = 122
    elif max(wnum) > 2800:
        logger.debug("I believe I'm working with ch2 data")
        foo = np.where(wnum >= 2550)
        minv = -0.2
        maxv = 2.0
    else:
        logger.error("Unable to determine the channel...")
        return

    good = np.where((minv <= rad[:, foo[0][0]]) & (rad[:, foo[0][0]] < maxv) & (qcflag == 0))
    logger.debug(f"There are {len(good[0])} good samples out of {len(secs)}")

    doplot=True
    if doplot:
        plt.plot(secs,rad[:,310],label='Original')
        plt.plot(secs[good[0]],rad[good[0],310],label='Flagged')
        plt.legend()
        plt.ylim(100,150)
        plt.ylabel('Radiance at '+str(round(wnum[310]))+' cm-1')
        plt.xlabel('Seconds')
        plt.title('Step: Create PC for period '+files[0].split('.')[1]+' to '+files[-1].split('.')[1])
        plt.savefig(os.path.join(tdir,'Create_PCA_radiance'+files[0].split('.')[1]+'_'+files[-1].split('.')[1]+'.png'))

    # If the number of good spectra is too small, then we should abort
    if len(good[0]) <= 3*len(wnum):
        logger.critical("There are TOO FEW good spectra for a good PCA noise filter (need > 3x at least")
        logger.critical("Aborting")
        return
    # If the keyword is set, use the median noise spectrum instead of the real noise spectrum
    if use_median_noise:
        logger.info("Using the median noise specturm, not the true sample noise spectrum")
        for i in range(len(nwnum)):
            nrad[:, i] = np.median(nrad[:, i])
    # Normalize the data
    import copy
    rad_orig=copy.deepcopy(rad)

    for i in range(len(secs)):
        rad[i, :] = rad[i, :] / np.interp(wnum, nwnum, nrad[i, :])
    # Generate the PCs
    pcwnum = wnum.copy()

    # Compute mean spectrum
    m = np.mean(rad[good], axis=0)

    c = np.cov(rad[good].T)

    logger.debug("Computing the SVD")
    u, d, v = svd(c)
    PC = {'u': u, 'd': d, 'm': m}
    # Determine the number of eigenvectors to use
    nvecs = pca_ind(d, len(good[0]), doplot=False)
    logger.info(f"Number of e-vecs used in reconstruction: {nvecs}")
    pca_comment = f"{len(d)} total PCs, {nvecs} considered significant, derived from {len(good[0])} " \
                    f"time samples, computed on {datetime.utcnow().isoformat()}"

    gsecs = secs[good]
    data = {'gsecs': gsecs, 'nvecs': nvecs, 'pcwnum': pcwnum, 'pca_comment': pca_comment,
            'PC': PC}
    with open(pcs_filename, 'wb') as fh:
        pickle.dump(data, fh)

    logger.info("DONE creating the PCs and storing them. The NF was not applied yet")

    return


def apply_irs_noise_filter(files, sdir, sfields, tdir, odir, pcs_filename=None, use_median_noise=False, sky_view_angle=None):
    """
    Abstract:
        This function is designed to apply the noise filter to IRS observations
        and place the noise-filtered data into the same netCDF file.  It uses
        a Principal Component Analysis technique published in Turner et al.
        JTECH 2006.  It is designed to use either ARM-formatted AERI, dmv-ncdf
        formatted AERI, or ASSIST data.  

    Author: Tyler Bell (CIWRO/NSSL, 2023) 2023
        Based on code from Dave Turner

    """
    
    logger.debug("Copying files to temporary directory...")
    new_files = []
    for fn in files:
        copy2(fn, tdir)
        #make sure that I am allowed to write to file
        os.chmod(os.path.join(tdir, os.path.basename(fn)),0o644)
        new_files.append(os.path.join(tdir, os.path.basename(fn)))
    # Read in the radiance data
    os.chdir(tdir)
    files = new_files

    # Read in the radiance data
    secs,rad,qcflag,wnum = read_ch_radiance_data(files,sky_view_angle)

    # Now read in the summary files
    nsecs,nrad,nqcflag,nwnum=read_sum_noise_data(secs,sdir,sfields,sky_view_angle)


    #Add ncqcflag to qcflag
    logger.debug(f"Merge QC flags for sum and ch1")
    cqcflag=np.zeros(qcflag.shape)
    for i in range(len(secs)):
        dels=np.abs(secs[i]-nsecs)
        #sum and ch1 samples need to be less than 5 secs apart
        foo=np.where((dels  == np.min(dels)) & (np.min(dels) < 5))[0]
        if len(foo) == 1:
            cqcflag[i]=nqcflag[foo] or qcflag[i]

    qcflag=cqcflag


    # Quick check for consistency
    logger.debug(f"Adding median noise value to missing summary samples and limit summary samples to time stamps of rad data")
    medianNoise=np.mean(nrad,axis=0)
    zrad=np.full((rad.shape[0],len(nwnum)),np.nan)
    zsecs=np.full((rad.shape[0]),np.nan)
    for i in range(len(secs)):
        dels=np.abs(secs[i]-nsecs)
        foo=np.where((dels  == np.min(dels)) & (np.min(dels) < 5))[0]
        if len(foo) == 1:
            #print('use actual noise value')
            zrad[i,:]=nrad[foo,:]
            zsecs[i]=nsecs[foo]
        else:
            print('use median noise')
            zrad[i,:]=medianNoise
            zsecs[i]=secs[i]

    nrad=zrad
    nsecs=zsecs
  

    osecs = secs
    orad = rad.transpose()
    onrad = nrad.transpose()
    oqc = qcflag
    
    logger.info(f"Number of samples: {len(secs)}; Number of spectral channels: {len(wnum)}")

    # Select only the good data (simple check)
    if min(wnum) < 600:
        logger.debug("I believe I'm working with ch1 data")
        foo = np.where(wnum >= 900)
        # We assume 30 deg C, which results in an upper threshold of 122RU, which is reasonable for many locations.
        minv = -2
        maxv = 122
    elif max(wnum) > 2800:
        logger.debug("I believe I'm working with ch2 data")
        foo = np.where(wnum >= 2550)
        minv = -0.2
        maxv = 2.0
    else:
        logger.error("Unable to determine the channel...")
        return

    good = np.where((minv <= rad[:, foo[0][0]]) & (rad[:, foo[0][0]] < maxv) & (qcflag == 0))
    logger.debug(f"There are {len(good[0])} good samples out of {len(secs)}")

    rad=rad[good]
    nrad=nrad[good]
    secs=secs[good]


    # If the number of good spectra is too small, then we should abort

    if len(good[0]) <= 3*len(wnum):
        logger.critical("There are TOO FEW good spectra for a good PCA noise filter (need > 3x at least")
        logger.critical("Aborting")
        os.chdir(tdir)
        for fn in files:
            os.remove(os.path.basename(fn))
        return

    # If the keyword is set, use the median noise spectrum instead of the real noise spectrum
    if use_median_noise:
        logger.info("Using the median noise specturm, not the true sample noise spectrum")
        for i in range(len(nwnum)):
            nrad[:, i] = np.median(nrad[:, i])
    # Normalize the data
    import copy
    rad_orig=copy.deepcopy(rad)
    for i in range(len(secs)):
        rad[i, :] = rad[i, :] / np.interp(wnum, nwnum, nrad[i, :])
    # Make sure the pickle file exists
    if not os.path.exists(pcs_filename):
        logger.error(f"The PCS file {pcs_filename} does not exist!")
        os.chdir(tdir)
        for fn in files:
            os.remove(os.path.basename(fn))
        return

    # Open the pickle file and load the information 
    logger.info("Restoring PCs from a file (independent PCA noise filtering)")
    with open(pcs_filename, 'rb') as fh:
        data = pickle.load(fh)

    gsecs = data['gsecs']
    nvecs = data['nvecs']
    pcwnum = data['pcwnum']
    pca_comment = data['pca_comment']
    PC = data['PC']

    logger.debug(f"Number of e-vecs used in reconstruction: {nvecs}")

    # Confirm the wavenumber array here matches that used to generate the PCs
    delta = np.abs(wnum - pcwnum)
    foo = np.where(delta > 0.001)
    if len(foo[0]) > 0:
        logger.error("The wavenumber array with the PCs does not match the current wnum array")
        return

    # Project the data onto this reduced basis
    logger.debug('Projecting the coefficients')
    coef = pca_project(rad, nvecs, PC)

    # Expand the data
    logger.debug("Expanding the reduced basis set")
    frad = rad
    feh = pca_expand(coef, PC)
    frad = feh
    # Remove the noise normalization
    for i in range(len(secs)):
        frad[i, :] = frad[i, :] * np.interp(wnum, nwnum, nrad[i, :])
        

    doplot=True
    if doplot:
        plt.plot(osecs,orad.T[:,310],label='Original')
        plt.plot(secs,rad_orig[:,310],label='Flagged')
        plt.plot(secs,frad[:,310],label='noise filtered')
        plt.legend()
        plt.ylim(100,150)
        plt.ylabel('Radiance at '+str(round(wnum[310]))+' cm-1')
        plt.xlabel('Seconds')
        plt.title('Step: Apply PC for period '+files[0].split('.')[1]+' to '+files[-1].split('.')[1])
        plt.savefig(os.path.join(tdir,'Apply_PCA_radiance'+files[0].split('.')[1]+'_'+files[-1].split('.')[1]+'.png'))


    # Now place the noise-filtered data back into the netCDF files, and copy them to the output directory
    # Only do this if the file does not already exist
    os.chdir(tdir)
    for fn in files:
        if os.path.isfile(os.path.join(odir,os.path.basename(fn))):
            logger.debug(f"Noise filtered file already exist in {os.path.join(odir, os.path.basename(fn))}")
            os.remove(fn)
            continue

        nc = Dataset(fn, 'a')
        rad = nc['mean_rad'][:]
        bt = nc['base_time'][0]
        
        if 'time_offset' in nc.variables.keys():
            to = nc['time_offset'][:]   
        elif 'time' in nc.variables.keys():
            bt *= 1e-3
            to = nc['time'][:]
        
        missingDataFlag = np.ones(to.shape)
         
        bt=bt.astype(np.float64)
        to=to.astype(np.float64)
        fsecs = bt+to
        kk = len(frad[0, :])
        for j, fsec in enumerate(fsecs):
            delta = np.abs(fsec - secs)
            foo = np.argmin(delta)
            if delta[foo] <  2:
                #Because I excluded the blackbody views and suspicious data before filtering, the timestamps are not the same anymore
                #I keep the original values when no noise filtered sample is available
                rad[j, 0:kk] = frad[foo, :]
                #Add missingDataFlag to the nc file (0: good, 1: flagged
                missingDataFlag[j]=0

        nc['mean_rad'][:] = rad
        nc.createVariable('missingDataFlag',np.int32,('time'))
        nc['missingDataFlag'][:]=missingDataFlag
        nc['missingDataFlag'].comment='Missing data flag created while doing the PCA noise filtering. 0: good, 1: flagged. Only good data were noise filtered. The flag includes stuck mirror, lossse in responsivity, spikes due to errors in mirror movement, hatch closed, zenith view.'  
        nc.setncattr('Noise_filter_comment', f'PCA noise filter was applied to the data, with {nvecs}'
                                             f' PCs used in the reconstruction')
        nc.setncattr('Noise_filter_comment2', pca_comment)
        nc.close()

        logger.debug(f"Final data in {os.path.join(odir, os.path.basename(fn))}")
        move(fn, os.path.join(odir, os.path.basename(fn)))


 #   #remove the remaining files from  tmp
 #   for fn in files[:-1]:
 #       print('remove '+fn)
 #       os.remove(fn)












