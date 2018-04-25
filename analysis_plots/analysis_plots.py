#!/usr/bin/python

import numpy as np
import sys
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as be_pdf

usage='''
Usage: analysis_plots.py <infile> <outdir>
where infile is the raw .fits file, and outdir is the directory to dump the PDF file to
'''

def compute_charge_masks(images, image_slice=None,sigma_estimate=30, sigma_factor=5):
  '''
  Masks anything above a threshold of sigma_estimage*sigma_factor
  You can pass in a tuple of slices of form (y,x) if you want to compute the mask only for
  a specific region
  '''
  if image_slice is None:
    image_slice=[]
    image_slice.append(slice(0,images.shape[2]))
    image_slice.append(slice(0,images.shape[3]))

  image_median=np.median(images[:,:,image_slice[0],image_slice[1]],axis=(2,3))
  threshold=sigma_estimate*sigma_factor
  mask=np.abs(images[:,:,image_slice[0],image_slice[1]]-image_median[:,:,np.newaxis,np.newaxis])<threshold
  return mask

def compute_dark_current(images,image_slice, overscan_slice):
  image_mask=compute_charge_masks(images,np.s_[:,image_slice[1]])
  overscan_mask=compute_charge_masks(images,np.s_[:,overscan_slice[1]])

  #Our mask has unmasked pixels "true", masked ones "false", numpy masks are the reverse
  image_masked=np.ma.array(images[:,:,:,image_slice[1]],mask=np.logical_not(image_mask))
  overscan_masked=np.ma.array(images[:,:,:,overscan_slice[1]],mask=np.logical_not(overscan_mask))
  
  image_averages_row=np.ma.average(image_masked,axis=3)
  overscan_averages_row=np.ma.average(overscan_masked,axis=3)  
  image_averages_row-=overscan_averages_row
  
  dark_current=np.ma.average(image_averages_row[:,:,image_slice[0]],axis=2)-np.ma.average(image_averages_row[:,:,overscan_slice[0]],axis=2)

  return dark_current.data #Return just the numpy array (the mask is now irrelevant



if __name__=="__main__":
  if len(sys.argv) < 3:
    print(usage)
    sys.exit(1)

  hdus=[]
  #Valid extensions
  extensions=[1,2,3,4,6,11,12]

  #Clean loops are done right before these images
  clean_loops=[3203,3250,3332,3417, 3453] 
  #Crashed occured right before this
  crashes=[0004,3345]
  
  nruns=1 #Code *can* load many runs, but currently interface limits to looking at one run at a time
  #Note: "run" here means a single image
  images=np.zeros(shape=(nruns,len(extensions),193,8544))

  runs=[]
  runs.append(sys.argv[1])
  out_directory=sys.argv[2]

  for run_number, fname in enumerate(runs):
    if not os.path.isfile(fname):
      print("Error, need to pass in a valid file name")
      sys.exit(2)

    hdus.append(fits.open(fname,memmap=True))
    for i, extension in enumerate(extensions):
      #Read the data from the HDU
      images[run_number,i,:,:]=hdus[run_number][extension].data

      
  masked_images=np.ma.array(images,mask=np.logical_not(compute_charge_masks(images)))
  
