#!/usr/bin/python


#########################################################################
# Simple scipt to take a raw 1x100 .fits file from DAMIC-100 and output #
# a) the dark current (estimated from the difference between average    #
# image pixel values and the average of the overscan region) and b) a   #
# PDF of the average of the image and overscan, both by row and by      #
# column                                                                #
#########################################################################


import numpy as np
import sys
import re
import os
import subprocess
from time import gmtime, strftime
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as be_pdf

usage='''Usage: analysis_plots.py <infile> <outdir> [<database_script>]
<infile>        : the raw .fits file
<outdir>        : the directory to dump the PDF file to
[<database_script>] : the location of the sendMonitInflux.py file to send dark current measurement to monitoring site
'''

send_to_database=False
send_to_wiki=False
#Constants that define image locations
#Should be moved into separate file?
x_overscan_slice=slice(8389,8538)
y_overscan_slice=slice(44,192)

y_image_slice=slice(2,43)
x_image_slice=slice(4273,8389)

right_overscan_slice=np.s_[y_image_slice,x_overscan_slice]

x_crop_CNS=400
right_image_DC_slice=np.s_[y_image_slice,4273+x_crop_CNS:8388-x_crop_CNS]


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
  '''
  Computes dark current by the following method:
  a) Mask off events using the standard charge mask
  b) Take the average by row of the image and x-overscan
  c) Subtract the x-overscan averages from the image
  d) Subtract the average of the image from the average of the y-overscan
  '''

  image_mask=compute_charge_masks(images,np.s_[image_slice[0],image_slice[1]])
  y_overscan_mask=compute_charge_masks(images,np.s_[overscan_slice[0],image_slice[1]])
  x_overscan_mask=compute_charge_masks(images,np.s_[:,overscan_slice[1]])

  #Our mask has unmasked pixels "true", masked ones "false", numpy masks are the reverse
  image_masked=np.ma.array(images[:,:,image_slice[0],image_slice[1]],mask=np.logical_not(image_mask))
  y_overscan_masked=np.ma.array(images[:,:,overscan_slice[0],image_slice[1]],mask=np.logical_not(y_overscan_mask))
  x_overscan_masked=np.ma.array(images[:,:,:,overscan_slice[1]],mask=np.logical_not(x_overscan_mask))
  
  image_averages_row=np.ma.average(image_masked,axis=3)
  x_overscan_averages_row=np.ma.average(x_overscan_masked,axis=3)  
  y_overscan_averages_row=np.ma.average(y_overscan_masked,axis=3)  

  image_averages_row-=x_overscan_averages_row[:,:,image_slice[0]]
  y_overscan_averages_row-=x_overscan_averages_row[:,:,overscan_slice[0]]

  image_averages=np.ma.average(image_averages_row,axis=2)
  overscan_averages=np.ma.average(y_overscan_averages_row,axis=2)
  dark_current=image_averages-overscan_averages

  return dark_current.data #Return just the numpy array (the mask is now irrelevant


if __name__=="__main__":

  if len(sys.argv) < 3:
    print(usage)
    sys.exit(1)

    
  hdus=[]
  #Valid extensions
  extensions=[1,2,3,4,6,11,12]
  
  nruns=1 #Code *can* load many runs, but currently interface limits to looking at one run at a time
  #Note: "run" here means a single image
  images=np.zeros(shape=(nruns,len(extensions),193,8544))

  fname=sys.argv[1]
  #While we're at it parse the path to the file
  fpaths=os.path.normpath(fname).split(os.path.sep)
  fpaths.reverse()
  #This assumes we follow the normal naming conventions

  runs=[]
  runs.append(fname)
  out_directory=sys.argv[2]
  
  for run_number, fname in enumerate(runs):
    if not os.path.isfile(fname):
      print("Error, need to pass in a valid file name")
      sys.exit(2)

    hdus.append(fits.open(fname,memmap=True))
    for i, extension in enumerate(extensions):
      #Read the data from the HDU
      images[run_number,i,:,:]=hdus[run_number][extension].data

  try:
    runID=int(hdus[0][0].header['RUNID'])
  except:
    print("Warning: runID not found in header, extracting from file name...")
    #Hacky but works (loads the file name in reverse, looks for 4 digits, then reverses those again to get the last group of four digits in the file name which should be the runID)
    runID=re.search("[0-9]{4}",fname[::-1]).group[::-1] 
    
  image_time=strftime("%Y-%m-%dT%H:%M:%SZ", gmtime(hdus[0][0].header['EXPSTOP']))
    
  charge_mask=np.zeros(images.shape, dtype=bool)
  #Only bother creating the mask for the image region, overscans rarely show charge
  charge_mask[:,:,y_image_slice,x_image_slice]=np.logical_not(compute_charge_masks(images,(y_image_slice,x_image_slice)))
  masked_images=np.ma.array(images,mask=charge_mask)
  #Computes dark current by run and by extension
  dark_current_by_run_ext=compute_dark_current(images,right_image_DC_slice,np.s_[y_overscan_slice,x_overscan_slice])
  print(dark_current_by_run_ext)
  #Averages by row
  x_overscan_average_rows_ext=np.ma.average(images[:,:,y_image_slice,x_overscan_slice],axis=(0,3))
  image_average_rows_ext=np.ma.average(images[:,:,y_image_slice,right_image_DC_slice[1]],axis=(0,3))

  #Averages by column
  y_overscan_average_columns_ext=np.ma.average(images[:,:,y_overscan_slice,x_image_slice],axis=(0,2))
  image_average_columns_ext=np.ma.average(images[:,:,y_image_slice,x_image_slice],axis=(0,2))

  #Labels for plotting
  temp=np.arange(1,9000)
  image_rows=temp[y_image_slice]
  image_columns=temp[x_image_slice]
  overscan_rows=temp[y_overscan_slice]
  overscan_columns=temp[x_overscan_slice]
  
  plt.ioff()
  
  figures=[]
  outfname=out_directory + "/plots_"+str(runID)+".pdf"
  pdf = be_pdf.PdfPages(outfname)

  for i, extension in enumerate(extensions):
    
    fig=plt.figure()
    plt.plot(image_columns,y_overscan_average_columns_ext[i],'r',label="Y overscan")
    plt.plot(image_columns,image_average_columns_ext[i],'b',label="Image")
    #plt.plot(image_columns, image_y_overscan_residual_ext[i], 'b', label="Residual")
    plt.title("Image and y_overscan by column for extension" +str(extension)+ " for runID " + str(runID))
    plt.xlabel("Column")
    plt.ylabel("Average pixel value")
    #Set range to a resonable value to exclude most charge
    ymax=max(np.percentile(y_overscan_average_columns_ext[i],90),np.percentile(image_average_columns_ext[i],90))
    ymin=min(np.percentile(y_overscan_average_columns_ext[i],5), np.percentile(image_average_columns_ext[i],5))
    plt.ylim(ymin=ymin-50,ymax=ymax+100)    
    plt.legend(loc='best')
    pdf.savefig(fig)
    
    fig=plt.figure()
    plt.plot(image_rows,x_overscan_average_rows_ext[i],'r',label="X overscan")
    plt.plot(image_rows,image_average_rows_ext[i],'b', label="Image")
    plt.title("Average of overscan/image by row for extension" +str(extension)+ " for runID " + str(runID))
    plt.xlabel("Row")
    plt.ylabel("Average pixel values")
    ymax=max(np.percentile(x_overscan_average_rows_ext[i],90), np.percentile(image_average_rows_ext[i],90))
    ymin=min(np.percentile(x_overscan_average_rows_ext[i],5), np.percentile(image_average_rows_ext[i],5))
    plt.ylim(ymin=ymin-50,ymax=ymax+100)
    plt.legend(loc='best')
    pdf.savefig(fig)
      
  pdf.close()

  if send_to_database:
    script=sys.argv[3]
    if not os.path.isfile(script):
      print("Error, database script not found")
    else:
      for i,extension in extensions:
        subprocess.call([script, 'snolab', 'dark current', dark_current_by_run_ext[0][i],'ext', extension])

  if send_to_wiki:
    pass
