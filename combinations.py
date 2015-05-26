# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:19:38 2015

@author: jselsing
"""

from matplotlib import rc_file
rc_file('/Users/jselsing/Pythonlibs/plotting/matplotlibstyle.rc')

import matplotlib.pylab as pl
# use seaborn for nice default plot settings
import seaborn; seaborn.set_style('ticks')

from astropy.io import fits
import glob
import numpy as np

    
def bin_image(img,binx,biny):
    import numpy as np
    from gen_methods import wmean
    
    """
    Used to bin low S/N 2D data from xshooter.
    Calculates the weigted averages of the bins, using the weighted average.
    Returns binned 2dimage
    """

    if binx == 1 and biny == 1:
        return img
    
#    ;--------------
    s=np.shape(img[0].data)

    outsizex=s[0]/binx

    outsizey=s[1]/biny

    res = np.zeros((outsizex,outsizey))
    reserr = np.zeros((outsizex,outsizey))
    bpmap = np.zeros((outsizex,outsizey))


    for i in np.arange(0,s[0]-(binx+1),binx):
         for j in np.arange(0,s[1]-(biny+1),biny):

#            w = 1./((img[1].data[i:i+binx,j:j+biny])**2.0)
            w = 1./((np.ones(np.shape(img[0].data))[i:i+binx,j:j+biny])**2.0)
#            w = np.ones(np.shape(img[0].data))
#            print w
            res[((i+binx)/binx-1),((j+biny)/biny-1)], reserr[((i+binx)/binx-1),((j+biny)/biny-1)] = wmean(img[0].data[i:i+binx,j:j+biny], w, axis = 1,  reterr = True)
#            bpmap[((i+binx)/binx-1),((j+biny)/biny-1)] = np.sum(img[2].data[i:i+binx,j:j+biny])

    img[0].header['NAXIS'] = 2
    img[0].header['NAXIS1'] = outsizey
    img[0].header['NAXIS2'] = outsizex
    img[0].header['CDELT1'] = img[0].header['CDELT1']*biny 
    img[0].header['CDELT2'] = img[0].header['CDELT2']*binx 
    img[0].header['CD2_2'] = img[0].header['CD2_2']*biny 
    img[0].header['CD1_1'] = img[0].header['CD1_1']*biny 
    img[0].data = res
#    img[1].data = reserr
#    img[2].data = bpmap  

    
    return img


#    from ian_astro_spec import spextractor
#
#    gain = [1.61, 1.67, 2,1]
#    ron = [2.6, 3.2, 8]
#    order_center = []
#    for i, a in enumerate(arms):
#        order_center.append(np.array(np.shape(files[a][0].data))/2.0)
#
#    onedspec_opt = []
#    for i, a in enumerate(arms):
#        print gain[i], ron[i], order_center[i]
#        onedspec_opt.append(spextractor(files[a][0].data, gain[i], ron[i], framevar=files[a][1].data, badpixelmask=files[a][2].data,
#                                        options=dict(bkg_radii=[20,30], extract_radius=15, bord=2, bsigma=3, pord=3, psigma=8, dispaxis=0, csigma=15, nreject=10000),  
#                                        trace_options=dict(ordlocs = np.array([order_center[1][i], order_center[0][i]])), verbose=True))
#    return onedspec_opt

#    from ian_astro_spec import spextractor
#
#    gain = 1.67
#    ron = 3.2
#
#
#    order_center = (np.array(np.shape(files[arm][0].data))/2.0)
#    
#
#
#    print gain, ron, order_center
#    bp_map = files[arm][2].data.copy()
#
#    bp_map[bp_map > 0] = 9999
#    bp_map[bp_map == 0] = 1
#    bp_map[bp_map == 9999] = 0
#    print bp_map
#
#    var = np.ones(np.shape(files[arm][2].data))
#
#    #print files[arm][1].data
#    #print files[arm][1].data**2
#    onedspec_opt = spextractor(files[arm][0].data, gain, ron, framevar=var, badpixelmask=bp_map,
#                                    options=dict(csigma=15, nreject=100),  
#                                    trace_options=dict(ordlocs = np.array([order_center[1], order_center[0]])), verbose=1)
#    return onedspec_opt

if __name__ == "__main__":

    img = glob.glob('/Users/jselsing/Work/X-Shooter/XSGRB/GRB230415A/xsh_scired_slit_stare_combined_SCI_SLIT_FLUX_MERGE2D_UVB_UVB_1x2_100k.fits')
    img = fits.open(img[0])
    wl = np.arange(img[0].header['CRVAL1'], img[0].header['CRVAL1'] + img[0].header['CDELT1']*img[0].header['NAXIS1'] ,img[0].header['CDELT1'])

#    print img
#    img = bin_image(img, 1, 3)
#    img.writeto('binned_image_uvb3.fits', clobber=True)
    
    img = glob.glob('/Users/jselsing/Work/X-Shooter/XSGRB/GRB230415A/xsh_scired_slit_stare_combined_SCI_SLIT_FLUX_MERGE2D_VIS_VIS_1x2_100k.fits')
    img = fits.open(img[0])
    
    
    from ian_astro_spec import spextractor

#    gain = 1.61
#    ron = 2.6

    gain = 1.67
    ron = 3.2
    order_center = (np.array(np.shape(img[0].data))/2.0)
    onedspec_opt = spextractor(img[0].data, gain, ron,
#                                        options=dict(bkg_radii=[20,30], extract_radius=15, bord=2, bsigma=3, pord=3, psigma=8, dispaxis=0, csigma=15, nreject=10000),  
                                        options=dict(dispaxis=0),                                         
                                        trace_options=dict(ordlocs = np.array([order_center[1], order_center[0]])), verbose=True)
    wl = np.arange(img[0].header['CRVAL1'], img[0].header['CRVAL1'] + img[0].header['CDELT1']*img[0].header['NAXIS1'] ,img[0].header['CDELT1'])

    pl.plot(binning1d(wl, 20), binning1d(onedspec_opt.spectrum, 20), lw = 0.3)
    pl.show()
    print onedspec_opt.varSpectrum

#    img = bin_image(img, 1, 3)
#    img.writeto('binned_image_vis3.fits', clobber=True)