#!/usr/bin/ipython -i

import numpy as np
from astropy.io import fits
import socket

import matplotlib.pyplot as plt

import os
import sys
import time

from scipy.stats import norm


#Attempt at prettier graphs, but the color schema doesn't work very well (extension 1 and 12 look identical)
# try:
#   import seaborn as sns
#   sns.set()
#   sns.set_style("white")
# except:
#   pass

#TODO: change all single slice to "slices", all double slices to "slices"
#TODO: implements this everywhere appropriate
class ImageShape:
  xcrop=400
  def __init__(self):
    self.set_variables()
  def set_variables(self,new_crop=None):
    if new_crop is not None:
      xcrop=new_crop

    self.right_overscan_y_slice=slice(8389,8538)
    self.right_overscan_y_slice=slice(44,192)
    self.right_image_y_slice=slice(1,43)
    self.right_image_x_slice=slice(4273,8389)

    self.left_overscan_x_slice=slice(6,157,-1)
    self.left_overscan_y_slice=right_overscan_y_slice
    self.left_image_x_slice=slice(4272,157,-1)
    self.left_image_y_slice=right_image_y_slice
    
    self.right_image_CNS_x_slice=slice(4273+xcrop,8389-xcrop)
    self.right_image_CNS_y_slice=right_image_y_slice
    
    self.left_image_CNS_x_slice=slice(4272-xcrop,157+xcrop,-1)
    self.left_image_CNS_y_slice=left_image_y_slice
    
    self.right_image_DC_y_slice=right_image_CNS_x_slice
    self.right_image_DC_x_slice=right_image_CNS_y_slice
  

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
  masked_a=np.ma.array(a,mask=np.logical_not(mask))
  masked_b=np.ma.array(b,mask=np.logical_not(mask))
  
  mean_a=np.ma.mean(masked_a)
  mean_b=np.ma.mean(masked_b)
  cov=np.ma.sum((masked_a-mean_a)*(masked_b-mean_b))/masked_a.count()
  return cov

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

#%%
def subtract_pedestal(images,n_col_fits=4,n_row_fits=4):
  ##NOT IMPLEMENTED##
  if not np.ma.is_masked(images):
    mask=compute_charge_masks(images)

#%%
def subtract_pedestal_XY(data, left_image_slice, right_image_slice, left_overscan_slice, right_overscan_slice):
  XY_overscan_right_avg=np.average(data[:,:,right_overscan_slice[0],right_overscan_slice[1]],axis=(2,3))
  XY_overscan_left_avg=np.average(data[:,:,left_overscan_slice[0],left_overscan_slice[1]],axis=(2,3))

  data[:,:,left_image_slice[0],left_image_slice[1]]-=XY_overscan_left_avg[:,:,np.newaxis,np.newaxis]
  data[:,:,right_image_slice[0],right_image_slice[1]]-=XY_overscan_right_avg[:,:,np.newaxis,np.newaxis]

  data[:,:,left_image_slice[0],left_image_slice[1]]-=np.median(data[:,:,left_image_slice[0],left_image_slice[1]],axis=((2,3)))[:,:,np.newaxis,np.newaxis]
  data[:,:,right_image_slice[0],right_image_slice[1]]-=np.median(data[:,:,right_image_slice[0],right_image_slice[1]],axis=((2,3)))[:,:,np.newaxis,np.newaxis]

  return data

#%%
def subtract_CNS(images,left_side, right_side):
  '''
  Data is a 4D matrix with [runs,extension,y,x]
  Note: the images should have pedestal subtracted to before doing CNS
  left_side,right_side should be a np.s_ slice that gives the left/right sides of the image
  '''
  nextensions=images.shape[1]
  nruns=images.shape[0]
  threshold=30*6. #Threshold for pixel values, exclude everything above this
  mask=compute_charge_masks(images,np.s_[:,:])
  masked_images=np.ma.array(images,mask=np.logical_not(mask))
  for run in range(nruns):
    for extension in range(nextensions):
      a=np.zeros(nextensions)
      RHS=np.zeros(nextensions)
      LHS=np.zeros((nextensions, nextensions))
      
      for i in range(nextensions):

        left_image1=masked_images[run,i,left_side[0],left_side[1]]
        right_image=masked_images[run,extension,right_side[0],right_side[1]]
        #Secondary mask
        weights=(right_image.flatten()-np.ma.mean(right_image))<threshold

        RHS[i]=np.ma.cov(left_image1.flatten(),right_image.flatten())[1,0]      

        for j in range(nextensions):
          left_image2=masked_images[run,j,left_side[0],left_side[1]]
          LHS[i,j]=np.ma.cov(left_image1.flatten(), left_image2.flatten())[1,0]

      a[:]=np.dot(np.linalg.inv(LHS),RHS)
      #print(a)                  
      for i in range(nextensions):
        masked_images[run,extension,right_side[0],right_side[1]]-=masked_images[run,i,left_side[0],left_side[1]]*a[i]
#%%
def compute_noise_single_image(image,image_slice, threshold=180):
  mask=compute_charge_masks(image[np.newaxis,np.newaxis],image_slice)[0,0]
  masked_image=np.ma.array(image[image_slice[0], image_slice[1]],mask=np.logical_not(mask))
  (mu, sigma) = norm.fit(masked_image.compressed())
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

#%%

if __name__=="__main__":
  hostname=socket.gethostname()
  hdus=[]
  extensions=[1,2,3,4,6,11,12]
  #extensions=[1,2,11,12]
  run_numbers=4
  start_runID=3205
  end_runID=3207 #Non-inclusive
  runs=np.arange(start_runID,end_runID)

  run3=[]
  run4=[]
  #For dark current measurement, 4/4/2018 --RT
  run3=[3159,3160, 3161, 4,5,6,7,8,9,10]
  #run3=[3160,3161]

  run4=[x for x in range(3200,3800)]
  run5=[x for x in range(3200,3800)]
  
  clean_loops=[3203,3250,3332,3417, 3453] #Clean loops are done right before these images
  crashes=[0004,3345]
  cooldowns=[3337]
  led_exposures=[3536]
  warmups=[3637,3654]
  
  runs=[]
  runs.extend(run3)
  runs.extend(run4)
  runs.extend(run5)
  runs=np.array(runs)
  
  run_numbers=[]
  run_numbers.extend(3*np.ones(len(run3),dtype=int))
  run_numbers.extend(4*np.ones(len(run4),dtype=int))
  run_numbers.extend(5*np.ones(len(run5),dtype=int))
  run_numbers=np.array(run_numbers)
  
  nruns=len(runs)
  images=np.zeros(shape=(nruns,len(extensions),193,8544))
    
  #TODO: move some of these into variables
  x_overscan_slice=slice(8389,8538)
  y_overscan_slice=slice(44,192)

  y_image_slice=slice(2,43)
  x_image_slice=slice(4273,8389)
  #x_image_slice=slice(4373,8000)

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
    if run_number<5:
      directory="/data/damic/snolab/raw/DAMIC_SNOLAB_RUN1/Jan2017/sci/140K/cryoOFF/30000s-IntW800-OS/1x100/run"+str(run_number)+"/"
    elif run_number==5:
      directory="/data/damic/snolab/raw/DAMIC_SNOLAB_RUN1/Jan2017/sci/135K/cryoOFF/30000s-IntW800-OS/1x100/run"+str(run_number)+"/"
    if "ryan" in hostname: #For running locally
      directory="/run/user/1000/gvfs/sftp:host=zev.uchicago.edu,user=ryant" + directory
    #TODO: change to allow for non-30000s runs
    fname=directory+"d44_snolab_Int-800_Exp-30000_"+str(run).zfill(4)+".fits.fz"
    if os.path.isfile(fname):
      
      valid_runs.append(idx)
      #hdus.append(fits.open("data/d44_snolab_Int-800_Exp-30000_301"+str(run)+".fits.fz"))
      hdus.append(fits.open(fname,memmap=True))
      for i,extension in enumerate(extensions):
        images[idx,i,:,:]=hdus[nvalid_runs][extension].data
      nvalid_runs+=1
  
  print("Images loaded")
  images=images[valid_runs]
  valid_runs=np.array(valid_runs)
  runs=runs[valid_runs]
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


  masked_images=np.ma.array(images,mask=np.logical_not(compute_charge_masks(images)))
  
  #Now compute the overscans  
  x_overscan_average_rows_ext=np.average(images[:,:,y_image_slice,x_overscan_slice],axis=(0,3))
  y_overscan_average_rows_ext=np.average(images[:,:,y_overscan_slice,x_image_slice],axis=(0,3))
  xy_overscan_average_rows_ext=np.average(images[:,:,y_overscan_slice,x_overscan_slice],axis=(0,3))   
  image_average_rows_ext=np.average(images[:,:,y_image_slice,x_image_slice],axis=(0,3))
  
  y_overscan_average_columns_ext=np.average(images[:,:,y_overscan_slice,x_image_slice],axis=(0,2))
  x_overscan_average_columns_ext=np.average(images[:,:,y_image_slice,x_overscan_slice],axis=(0,2))
  xy_overscan_average_columns_ext=np.average(images[:,:,y_overscan_slice,x_overscan_slice],axis=(0,2))
  image_average_columns_ext=np.average(images[:,:,y_image_slice,x_image_slice],axis=(0,2))

  x_overscan_average_rows_run=np.average(images[:,:,y_image_slice,x_overscan_slice],axis=(1,3))
  y_overscan_average_rows_run=np.average(images[:,:,y_overscan_slice,x_image_slice],axis=(1,3))
  xy_overscan_average_rows_run=np.average(images[:,:,y_overscan_slice,x_overscan_slice],axis=(1,3))   
  image_average_rows_run=np.average(images[:,:,y_image_slice,x_image_slice],axis=(1,3))
  
  y_overscan_average_columns_run=np.average(images[:,:,y_overscan_slice,x_image_slice],axis=(1,2))
  x_overscan_average_columns_run=np.average(images[:,:,y_image_slice,x_overscan_slice],axis=(1,2))
  xy_overscan_average_columns_run=np.average(images[:,:,y_overscan_slice,x_overscan_slice],axis=(1,2))
  image_average_columns_run=np.average(images[:,:,y_image_slice,x_image_slice],axis=(1,2))

  masked_y_overscan_average_columns_run=np.ma.average(masked_images[:,:,y_overscan_slice,x_image_slice],axis=(1,2))
  masked_image_average_columns_run=np.ma.average(masked_images[:,:,y_image_slice,x_image_slice],axis=(1,2))
  
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

  dark_current_by_run_ext=compute_dark_current(images,right_image_DC_slice,np.s_[y_overscan_slice,x_overscan_slice])
  
  # plt.figure()
  # plt.plot(image_columns,np.ma.average(masked_y_overscan_average_columns_run,axis=0),'r',label="Y overscan")
  # plt.plot(image_columns,np.ma.average(masked_image_average_columns_run,axis=0),'b',label="Image")
  # plt.title("Average of overscan/image by column (all extensions)\n For runIDs "+ str(start_runID)+"-"+str(end_runID-1))
  # plt.xlabel("Column")
  # plt.ylabel("Average pixel value")
  # plt.legend(loc='best')
  
  # plt.figure()
  # plt.plot(image_rows,np.average(x_overscan_average_rows_run,axis=0),'r',label="X overscan")
  # plt.plot(image_rows,np.average(image_average_rows_run,axis=0),'b', label="Image")
  # plt.title("Average of overscan/image by row (all extensions)\n For runIDs "+ str(start_runID)+"-"+str(end_runID-1))
  # plt.xlabel("Row")
  # plt.ylabel("Average pixel values")
  # plt.legend(loc='best')
  
  # plt.figure()
  # plt.plot(overscan_rows,np.average(y_overscan_average_rows_ext,axis=0),'b', label="Y overscan")
  # plt.plot(overscan_rows,np.average(xy_overscan_average_rows_ext,axis=0),'r',label="XY overscan")
  # plt.title("Average of overscan by row")
  # plt.xlabel("Row")
  # plt.ylabel("Average pixel values")
  # plt.legend(loc='best')

  # plt.figure()
  # plt.plot(overscan_columns,x_overscan_average_columns,'b',label="X overscan")
  # plt.plot(overscan_columns,xy_overscan_average_columns,'r',label="XY overscan")
  # plt.legend(loc='best')
  # plt.title("Average of overscan by column")
  # plt.xlabel("Column")
  # plt.ylabel("Average pixel values")

  # plt.figure()
  # plt.plot(runs,dark_current_by_run,'D',label="Dark Current")
  # plt.title("Dark Current by run from runID " + str(start_runID) + " to " +str(end_runID))
  # plt.xlabel("RunID")
  # plt.ylabel("Dark Current (ADU)")
  # plt.legend(loc='best')

  ##--------------------Dark Current Plotting-----------------------##

  plt.figure()
  x=range(len(valid_runs))
  lines=plt.plot(x,dark_current_by_run_ext, "d")
  plt.title("Dark Current by RunID")
  plt.xlabel("RunID")
  plt.ylabel("DC (ADU)")
  plt.xticks(x,[str(y).zfill(4) for y in runs],rotation=-90)
  labels=["Ext 1", "Ext 2", "Ext 3", "Ext 4", "Ext 6", "Ext 11", "Ext 12"]
  for clean in clean_loops:
    if clean in runs:
      #Stupid bullshit hack because x[runs==clean] doesn't wanna work on Zev
      line_x=x[np.argwhere(runs==clean)[0][0]]-.5
      #Check if the line is interesting
      if (line_x>0):
        line=plt.axvline(line_x,color='b',linewidth=2, linestyle='dashed')
        #Labeling for the vertical line (only add one)
        if "Clean Loop" not in labels:
          lines.append(line)
          labels.append("Clean Loop")

  for crash in crashes:
    if crash in runs:
      line_x=x[np.argwhere(runs==crash)[0][0]]-.5
      line=plt.axvline(line_x,color='r',linewidth=2, linestyle='dashed')
      if "Crash" not in labels:
        lines.append(line)
        labels.append("Crash")

  #Cooldown run
  for cooldown in cooldowns:
    if cooldown in runs:
      line_x=x[np.argwhere(runs==cooldown)[0][0]]-.5
      line=plt.axvline(line_x,color='g',linewidth=2, linestyle='dashed')
      if "Cooldown" not in labels:
        lines.append(line)
        labels.append("Cooldown")
  # if 3337 in runs:
  #   line_x=x[np.argwhere(runs==3337)[0][0]]-.5
  #   line=plt.axvline(line_x,color='g',linewidth=2, linestyle='dashed')
  #   lines.append(line)
  #   labels.append("Cooldown")
  for led_exposure in led_exposures:
    if led_exposure in runs:
      line_x=x[np.argwhere(runs==led_exposure)[0][0]]-.5
      line=plt.axvline(line_x,color='m',linewidth=2, linestyle='dashed')
      if "LED Exposure "not in labels:
        lines.append(line)
        labels.append("LED Exposure")

  for warmup in warmups:
    if warmup in runs:
      line_x=x[np.argwhere(runs==warmup)[0][0]]-.5
      line=plt.axvline(line_x,color='c',linewidth=2, linestyle='dashed')
      if "Temp Cycle"not in labels:
        lines.append(line)
        labels.append("Temp Cycle")

  plt.ylim(ymin=0)
  plt.yscale('log')
  plt.legend(lines,labels ,loc='best')
  
  #Show labels on both sides
  ax=plt.gca()
  ax.tick_params(labelright=True)
  ax.yaxis.grid(True)
  if len(x) > 30:
    #Only show every other label (too cluttered otherwise)
    plt.setp(ax.get_xticklabels()[1::2],visible=False) 
  
  # for i, extension in enumerate(extensions):
    
  #   plt.figure()
  #   plt.plot(image_columns,y_overscan_average_columns_ext[i],'r',label="Y overscan")
  #   plt.plot(image_columns,image_average_columns_ext[i],'b',label="Image")
  #   #plt.plot(image_columns, image_y_overscan_residual_ext[i], 'b', label="Residual")
  #   plt.title("Image and y_overscan by column for extension" +str(extension))
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
  plt.show(False)
  
