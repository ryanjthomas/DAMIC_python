#!/usr/bin/python

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy.stats import norm

#TODO: change all single slice to "slices", all double slices to "slices"
#TODO: implements this everywhere appropriate
class ImageShape:
  xcrop=400
  right_overscan_y_slice=slice(8389,8538)
  right_overscan_y_slice=slice(44,192)
  right_image_y_slice=slice(1,43)
  right_image_x_slice=slice(4273,8389)

  left_overscan_x_slice=slice(6,157,-1)
  left_overscan_y_slice=right_overscan_y_slice
  left_image_x_slice=slice(4272,157,-1)
  left_image_y_slice=right_image_y_slice

  right_image_CNS_x_slice=slice(4273+xcrop,8389-xcrop)
  right_image_CNS_y_slice=right_image_y_slice
  
  left_image_CNS_x_slice=slice(4272-xcrop,157+xcrop,-1)
  left_image_CNS_y_slice=left_image_y_slice

  right_image_DC_y_slice=right_image_CNS_x_slice
  right_image_DC_x_slice=right_image_CNS_y_slice
  

def save_image(image, fname):
  hdu = fits.PrimaryHDU(image)
  hdulist = fits.HDUList([hdu])
  hdulist.writeto(fname + ".fits")

def compute_cov(a,b, threshold=None):
  #Covariance computation following recon CNS subtraction method
  if threshold is not None:
    mask=(a < threshold) * (b < threshold)
  else:
    mask=1.
  mean_a=np.mean(a)
  mean_b=np.mean(b)
  cov=np.sum((a*mask-mean_a)*(b*mask-mean_b))/len(a*mask)
  return cov
  
#%%
def subtract_pedestal_XY(data, left_image_slice, right_image_slice, left_overscan_slice, right_overscan_slice):
  XY_overscan_right_avg=np.average(data[:,:,right_overscan_slice[0],right_overscan_slice[1]],axis=(2,3))
  XY_overscan_left_avg=np.average(data[:,:,left_overscan_slice[0],left_overscan_slice[1]],axis=(2,3))

  data[:,:,left_image_slice[0],left_image_slice[1]]-=XY_overscan_left_avg[:,:,np.newaxis,np.newaxis]
  data[:,:,right_image_slice[0],right_image_slice[1]]-=XY_overscan_right_avg[:,:,np.newaxis,np.newaxis]

#%%
def subtract_CNS(data,left_side, right_side):
  '''
  Data is a 4D matrix with [runs,extension,y,x]
  Note: the images should have pedestal subtracted to before doing CNS
  left_side,right_side should be a np.s_ slice that gives the left/right sides of the image
  '''
  nextensions=data.shape[1]
  nruns=data.shape[0]
  threshold=30*6. #Threshold for pixel values, exclude everything above this
  for run in range(nruns):
    for extension in range(nextensions):
      a=np.zeros(nextensions)
      RHS=np.zeros(nextensions)
      LHS=np.zeros((nextensions, nextensions))

      for i in range(nextensions):

        left_image1=data[run,i,left_side[0],left_side[1]]
        right_image=data[run,extension,right_side[0],right_side[1]]
        weights=(right_image.flatten()-np.mean(right_image))<threshold
        RHS[i]=np.cov(left_image1.flatten()[weights],right_image.flatten()[weights])[1,0]      
        for j in range(nextensions):
          left_image2=data[run,j,left_side[0],left_side[1]]
          LHS[i,j]=np.cov(left_image1.flatten(), left_image2.flatten())[1,0]

      a[:]=np.dot(RHS,np.linalg.inv(LHS))
      #print(a)
      for i in range(nextensions):
        data[run,extension,right_side[0],right_side[1]]-=data[run,i,left_side[0],left_side[1]]*a[i]
   
#%%
def compute_noise_single_image(image,image_slice, threshold=180):
  mask=image[image_slice[0],image_slice[1]].flatten()<threshold
  (mu, sigma) = norm.fit(image[image_slice[0],image_slice[1]].flatten()*mask)
  return mu, sigma

def compute_noise_many_image(images,image_slice, threshold=180):
  nruns=images.shape[0]
  nextensions=images.shape[1]
  mus=np.zeros((nruns,nextensions))
  noises=np.zeros((nruns,nextensions))
  for run in range(nruns):
    for extension in range(nextensions):
      mu,noise=compute_noise_single_image(images[run,extension], image_slice, threshold)
      noises[run,extension]=noise
      mus[run,extension]=mu
  return mus,noises

#%%
def compute_charge_masks(images, image_slice,sigma_estimate=30, sigma_factor=5):
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

#%%

if __name__=="__main__":
  hdus=[]
  extensions=[1,2,3,4,6,11,12]
  #extensions=[1,2,11,12]
  run_numbers=4
  start_runID=3205
  end_runID=3207 #Non-inclusive
  runs=np.arange(start_runID,end_runID)


  #For dark current measurement, 4/4/2018 --RT
  #run3=[3160, 3161, 4,5,6,7,8,9,10]
  run3=[]
  run4=[x for x in range(3203,3217)]
  run4.extend([3250,3251,3252,3253,3254,3255,3256])
  
  runs=[]
  runs.extend(run3)
  runs.extend(run4)
  runs=np.array(runs)
  
  run_numbers=[]
  run_numbers.extend(3*np.ones(len(run3),dtype=int))
  run_numbers.extend(4*np.ones(len(run4),dtype=int))
  run_numbers=np.array(run_numbers)
  
  nruns=len(runs)
  images=np.zeros(shape=(nruns,len(extensions),193,8544))
    
  #TODO: move some of these into variables
  x_overscan_slice=slice(8389,8538)
  y_overscan_slice=slice(44,192)

  y_image_slice=slice(2,43)
  #x_image=slice(4273,8389)
  x_image_slice=slice(4373,8000)

  left_overscan_slice=np.s_[y_image_slice,6:157]
  right_overscan_slice=np.s_[y_image_slice,x_overscan_slice]
  
  left_image_slice=np.s_[1:193,4272:157:-1]
  right_image_slice=np.s_[1:193,4273:8389]
  
  #crops for the CNS subtraction  
  x_crop_CNS=400
  left_image_CNS_slice=np.s_[1:192,4272-x_crop_CNS:157+x_crop_CNS:-1]
  right_image_CNS_slice=np.s_[1:192,4273+x_crop_CNS:8388-x_crop_CNS]

  left_image_DC_slice=np.s_[y_image_slice,4272-x_crop_CNS:157+x_crop_CNS:-1]
  right_image_DC_slice=np.s_[y_image_slice,4273+x_crop_CNS:8388-x_crop_CNS]

  
  temp=np.arange(1,9000)
  image_rows=temp[y_image_slice]
  image_columns=temp[x_image_slice]
  overscan_rows=temp[y_overscan_slice]
  overscan_columns=temp[x_overscan_slice]

  valid_runs=[]
  start_time=time.time()

  #Convert our run_numbers from an integer to array
  nvalid_runs=0
  if type(run_numbers) is int:
    run_numbers=run_numbers*np.ones(nruns)
  #  for run in runs:
  for idx, (run, run_number) in enumerate(zip(runs, run_numbers)):
    directory="/run/user/1000/gvfs/sftp:host=zev.uchicago.edu,user=ryant/data/damic/snolab/raw/DAMIC_SNOLAB_RUN1/Jan2017/sci/140K/cryoOFF/30000s-IntW800-OS/1x100/run"+str(run_number)+"/"
    #directory="data/"
    fname=directory+"d44_snolab_Int-800_Exp-30000_"+str(run).zfill(4)+".fits.fz"
    if os.path.isfile(fname):
      
      valid_runs.append(idx)
      #hdus.append(fits.open("data/d44_snolab_Int-800_Exp-30000_301"+str(run)+".fits.fz"))
      hdus.append(fits.open(fname,memmap=True))
      for i,extension in enumerate(extensions):
        images[idx,i,:,:]=hdus[nvalid_runs][extension].data
      nvalid_runs+=1
  
  print("Images loaded")
  valid_runs=np.array(valid_runs)
  if len(valid_runs) <=0:
    print("error, no valid images found")
    sys.exit()
  #Run CNS  
  load_time=time.time()
  print("Time to load files is: " + str(load_time-start_time))

  # subtract_pedestal_XY(images,left_image_slice,right_image_slice, left_overscan_slice, right_overscan_slice)
  # pedestal_subtraction_time=time.time()
  # print("Time to subtract pedestal is: " + str(pedestal_subtraction_time-load_time))

  # _,noise_pre=compute_noise_many_image(images,right_image_CNS_slice)
  # print("Pre subtraction noise is: " + str(noise_pre))
  
  # subtract_CNS(images,left_image_CNS_slice, right_image_CNS_slice)
  # CNS_subtraction_time=time.time()
  # print("Time to subtract CNS is: " + str(CNS_subtraction_time-pedestal_subtraction_time))

  # _,noise_post=compute_noise_many_image(images,right_image_CNS_slice)
  # print("Post subtraction noise is: " + str(noise_post))

  
  #Now compute the overscans  
  x_overscan_average_rows_ext=np.average(images[valid_runs,:,y_image_slice,x_overscan_slice],axis=(0,3))
  y_overscan_average_rows_ext=np.average(images[valid_runs,:,y_overscan_slice,x_image_slice],axis=(0,3))
  xy_overscan_average_rows_ext=np.average(images[valid_runs,:,y_overscan_slice,x_overscan_slice],axis=(0,3))   
  image_average_rows_ext=np.average(images[valid_runs,:,y_image_slice,x_image_slice],axis=(0,3))
  
  y_overscan_average_columns_ext=np.average(images[valid_runs,:,y_overscan_slice,x_image_slice],axis=(0,2))
  x_overscan_average_columns_ext=np.average(images[valid_runs,:,y_image_slice,x_overscan_slice],axis=(0,2))
  xy_overscan_average_columns_ext=np.average(images[valid_runs,:,y_overscan_slice,x_overscan_slice],axis=(0,2))
  image_average_columns_ext=np.average(images[valid_runs,:,y_image_slice,x_image_slice],axis=(0,2))

  x_overscan_average_rows_run=np.average(images[valid_runs,:,y_image_slice,x_overscan_slice],axis=(1,3))
  y_overscan_average_rows_run=np.average(images[valid_runs,:,y_overscan_slice,x_image_slice],axis=(1,3))
  xy_overscan_average_rows_run=np.average(images[valid_runs,:,y_overscan_slice,x_overscan_slice],axis=(1,3))   
  image_average_rows_run=np.average(images[valid_runs,:,y_image_slice,x_image_slice],axis=(1,3))
  
  y_overscan_average_columns_run=np.average(images[valid_runs,:,y_overscan_slice,x_image_slice],axis=(1,2))
  x_overscan_average_columns_run=np.average(images[valid_runs,:,y_image_slice,x_overscan_slice],axis=(1,2))
  xy_overscan_average_columns_run=np.average(images[valid_runs,:,y_overscan_slice,x_overscan_slice],axis=(1,2))
  image_average_columns_run=np.average(images[valid_runs,:,y_image_slice,x_image_slice],axis=(1,2))

  #Now compute the residuals
  image_y_overscan_residual_ext=image_average_columns_ext-y_overscan_average_columns_ext
  image_x_overscan_residual_ext=image_average_rows_ext-x_overscan_average_rows_ext

  xy_x_overscan_residual_ext=xy_overscan_average_columns_ext-x_overscan_average_columns_ext
  xy_y_overscan_residual_ext=xy_overscan_average_rows_ext-y_overscan_average_rows_ext

  image_y_overscan_residual_run=image_average_columns_run-y_overscan_average_columns_run
  image_x_overscan_residual_run=image_average_rows_run-x_overscan_average_rows_run

  xy_x_overscan_residual_run=xy_overscan_average_columns_run-x_overscan_average_columns_run
  xy_y_overscan_residual_run=xy_overscan_average_rows_run-y_overscan_average_rows_run

  dark_current_by_run=(np.average(image_average_rows_run - np.average(y_overscan_average_rows_run,axis=1)[:,np.newaxis],axis=1)-
                       np.average(y_overscan_average_rows_run - np.average(y_overscan_average_rows_run,axis=1)[:,np.newaxis],axis=1))

  dark_current_by_run_ext=compute_dark_current(images[valid_runs],right_image_DC_slice,np.s_[y_overscan_slice,x_overscan_slice])
  
  plt.figure()
  plt.plot(image_columns,np.average(y_overscan_average_columns_run,axis=0),'r',label="Y overscan")
  plt.plot(image_columns,np.average(image_average_columns_run,axis=0),'b',label="Image")
  plt.title("Average of overscan/image by column (all extensions)\n For runIDs "+ str(start_runID)+"-"+str(end_runID-1))
  plt.xlabel("Column")
  plt.ylabel("Average pixel value")
  plt.legend(loc='best')
  
  plt.figure()
  plt.plot(image_rows,np.average(x_overscan_average_rows_run,axis=0),'r',label="X overscan")
  plt.plot(image_rows,np.average(image_average_rows_run,axis=0),'b', label="Image")
  plt.title("Average of overscan/image by row (all extensions)\n For runIDs "+ str(start_runID)+"-"+str(end_runID-1))
  plt.xlabel("Row")
  plt.ylabel("Average pixel values")
  plt.legend(loc='best')

  plt.figure()
  plt.plot(runs[valid_runs],dark_current_by_run,'D',label="Dark Current")
  plt.title("Dark Current by run from runID " + str(start_runID) + " to " +str(end_runID))
  plt.xlabel("RunID")
  plt.ylabel("Dark Current (ADU)")
  plt.legend(loc='best')
  
  plt.figure()
  plt.plot(overscan_rows,np.average(y_overscan_average_rows_ext,axis=0),'b', label="Y overscan")
  plt.plot(overscan_rows,np.average(xy_overscan_average_rows_ext,axis=0),'r',label="XY overscan")
  plt.title("Average of overscan by row")
  plt.xlabel("Row")
  plt.ylabel("Average pixel values")
  plt.legend(loc='best')

  plt.figure()
  x=range(len(valid_runs))
  lines=plt.plot(x,dark_current_by_run_ext, "d")
  plt.title("Dark Current by RunID")
  plt.xlabel("RunID")
  plt.ylabel("DC (ADU)")
  plt.xticks(x,(str(y).zfill(4) for y in runs[valid_runs]))
  plt.legend(lines, ["Ext 1", "Ext 2", "Ext 3", "Ext 4", "Ext 6", "Ext 11", "Ext 12"],loc='best')
  
  # plt.figure()
  # plt.plot(overscan_columns,x_overscan_average_columns,'b',label="X overscan")
  # plt.plot(overscan_columns,xy_overscan_average_columns,'r',label="XY overscan")
  # plt.legend(loc='best')
  # plt.title("Average of overscan by column")
  # plt.xlabel("Column")
  # plt.ylabel("Average pixel values")

  # for i, extension in enumerate(extensions):
  #   plt.figure()
  #   plt.plot(image_columns,y_overscan_average_columns_ext[i],'r',label="Y overscan")
  #   plt.plot(image_columns,image_average_columns_ext[i],'b',label="Image")
  #   #plt.plot(image_columns, image_y_overscan_residual_ext[i], 'b', label="Residual")
  #   plt.title("Residual of image-y_overscan by column for extension" +str(extension))
  #   plt.xlabel("Column")
  #   plt.ylabel("Average pixel value")
  #   plt.legend(loc='best')
   
  #   plt.figure()
  #   plt.plot(image_rows,x_overscan_average_rows_ext[i],'r',label="X overscan")
  #   plt.plot(image_rows,image_average_rows_ext[i],'b', label="Image")
  #   plt.title("Average of overscan/image by row for extension" +str(extension))
  #   plt.xlabel("Row")
  #   plt.ylabel("Average pixel values")
  #   plt.legend(loc='best')
  
#    plt.figure()
#    plt.plot(overscan_rows,y_overscan_average_rows_ext[i],'b', label="Y overscan")
#    plt.plot(overscan_rows,xy_overscan_average_rows_ext[i],'r',label="XY overscan")
#    plt.title("Average of overscan by row for extension" +str(extension))
#    plt.xlabel("Row")
#    plt.ylabel("Average pixel values")
#    plt.legend(loc='best')
#
#    plt.figure()
#    plt.plot(overscan_columns,x_overscan_average_columns_ext[i],'b',label="X overscan")
#    plt.plot(overscan_columns,xy_overscan_average_columns_ext[i],'r',label="XY overscan")
#    plt.legend(loc='best')
#    plt.title("Average of overscan by column for extension" +str(extension))
#    plt.xlabel("Column")
#    plt.ylabel("Average pixel values")
#    
  
