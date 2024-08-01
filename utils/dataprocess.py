"""
Coded by Chaochao Zhou, PhD (chaochao.zhou@northwestern.edu / czhouphd@gmail.com)
Radiology Department, Northwestern University and Medicine, Chicago, IL
"""

import numpy as np
import matplotlib.pyplot as plt
import neurite as ne
import voxelmorph as vxm
import tensorflow as tf
import os, sys, shutil
import nibabel as nib
import cv2

import ipywidgets
import bokeh
import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()


def load_xray_seq(folders, i_seq=None, glob_norm=False, filename='input.nii'):
    if i_seq is None:
        i_seq = np.random.choice(len(folders))

    path = os.path.join(folders[i_seq], filename)
    xrays_raw = nib.load(path).get_fdata()

    xrays = np.moveaxis(xrays_raw, -1, 0)
    if glob_norm:
        xrays = xrays / 4095.
    else:
        xrays = (xrays - xrays.min()) / (xrays.max() - xrays.min())
    # xrays = [cv2.resize(xa, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for xa in xrays]
    # xrays = [cv2.cvtColor(xa, cv2.COLOR_GRAY2RGB) for xa in xrays]
    xrays = np.array(xrays, dtype='float32')
    xrays = xrays[..., None]

    return xrays


def vxm_seq_generator(folders, i_seq=None, nimgs=None, replace=False, bidir=False):
  xrays = load_xray_seq(folders, i_seq=i_seq)
  n_all = len(xrays)

  if nimgs is None:
    nimgs = n_all
  else:
    if nimgs > n_all and replace == False:
      nimgs = n_all

  img_shape = xrays.shape[1:3]
  ndims = len(img_shape)
  zero_warps = np.zeros([nimgs, *img_shape, ndims], dtype='float32')

  idx = np.random.randint(3)
  images_moving = np.repeat(xrays[idx][None], nimgs, axis=0)

  indices = np.random.choice(a=n_all, size=nimgs, replace=replace)
  images_fixed = xrays[indices]

  inputs = [images_moving, images_fixed]
  if bidir:
  	outputs = [images_fixed, images_moving, zero_warps]
  else:
  	outputs = [images_fixed, zero_warps]
  return inputs, outputs


def hpm_seq_generator(folders, i_seq=None, nimgs=None, replace=False, nhps=1, bidir=False):
  inputs, outputs = vxm_seq_generator(folders=folders,
                                      i_seq=i_seq,
                                      nimgs=nimgs,
                                      replace=replace,
                                      bidir=bidir)
  nimgs = len(inputs[0])  # update nimgs
  hyperparams = np.random.rand(nimgs, nhps)
  inputs = [*inputs, hyperparams]

  return inputs, outputs
  

def plot_history(seq_history, global_history=None, figsize=(12, 3)):
  plt.figure(figsize=figsize, constrained_layout=True)

  plt.subplot(121)
  for loss_name in seq_history.history.keys():
    plt.plot(seq_history.epoch, seq_history.history[loss_name], '.-', label=loss_name)
  plt.title('sequence history')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  # plt.yscale('log')
  plt.legend()

  if global_history is not None:
    plt.subplot(122)
    for loss_name in global_history.keys():
      if loss_name is not 'iter':
        plt.plot(global_history['iter'], global_history[loss_name], label=loss_name)
    plt.title('learning curve')
    plt.ylabel('loss')
    plt.xlabel('iter')
    # plt.yscale('log')
    plt.legend()

  plt.show()


def plot_dsa_predictions(model, folders, i_seq, nhps=1, idx_bg=0, Ndisp=6, sub_width=3):
  xrays = load_xray_seq(folders=folders, i_seq=i_seq)
  image_bg = xrays[idx_bg][None, ...]

  indices = np.linspace(0, xrays.shape[0]-1, Ndisp).astype(np.int64)
  xrays = xrays[indices]  # extract a subset of xray sequence frames to display

  images_moving = np.repeat(image_bg, len(xrays), axis=0)
  images_fixed = xrays
  
  if nhps>=1:
  	hyperparams = np.random.rand(len(xrays), nhps)
  	preds = model.predict([images_moving, images_fixed, hyperparams], verbose=0)
  else:
  	preds = model.predict([images_moving, images_fixed], verbose=0)
  images_pred = preds[0]
  warps_pred = preds[-1]

  dsa_org = images_fixed - images_moving
  dsa_pred = images_fixed - images_pred

  dsa_org = (dsa_org - dsa_org.min()) / (dsa_org.max() - dsa_org.min())
  dsa_pred = (dsa_pred - dsa_pred.min()) / (dsa_pred.max() - dsa_pred.min())

  # fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*3,4), constrained_layout=True)
  fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*sub_width, sub_width))
  for i in range(Ndisp):
    axes[i].imshow(dsa_org[i], cmap='gray', vmin=np.min(dsa_org), vmax=np.max(dsa_org))
    axes[i].axis('off')
    axes[i].set_title('frm: {}'.format(indices[i]))
  plt.tight_layout()
  plt.show()

  # fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*3,4), constrained_layout=True)
  fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*sub_width, sub_width))
  for i in range(Ndisp):
    axes[i].imshow(dsa_pred[i], cmap='gray', vmin=np.min(dsa_pred), vmax=np.max(dsa_pred))
    axes[i].axis('off')
    if nhps == 1:
    	axes[i].set_title('λ: {:.2f}'.format(hyperparams[i,0]))
    elif nhps == 2:
    	axes[i].set_title('λ: {:.2f} | α: {:.2f}'.format(hyperparams[i,0], hyperparams[i,1]))
  plt.tight_layout()
  plt.show()

  flows = [warp[::6,::6,:] for warp in warps_pred]
  ne.plot.flow(flows, width=Ndisp*sub_width);


def run_interactive_search_1p(model, test_data, hp0=0.5):
    ## configure initial image pair to align
    moving_id = 0
    global fixed_id
    fixed_id = len(test_data)//2

    ## run the first model prediction, which tends to be slower
    ## since tensorflow needs to initialize some extra elements
    inputs = [test_data[moving_id][None], test_data[fixed_id][None], np.array([[hp0]])]
    predicted = model.predict(inputs)[0].squeeze()

    ## initialize figures for plotting
    figs = [bokeh.plotting.figure(title=t) for t in ['DSA-BG', 'DSA-Pred']]

    ## utilities for plotting image figures
    scale = lambda im: (im - im.min()) / (im.max() - im.min())
    mapper = bokeh.models.LinearColorMapper(palette='Greys256', low=0, high=255)
    convert = lambda im: np.flip(im * 255, 0).astype('uint8')
    plot = lambda id, im: figs[id].image(image=[convert(im)],
                                         x=0, y=0,
                                         dw=im.shape[0], dh=im.shape[1],
                                         color_mapper=mapper)

    ## plot the initial images
    dsa_bg = scale(test_data[fixed_id] - test_data[moving_id]).squeeze()
    dsa_pred = scale(test_data[fixed_id].squeeze() - predicted)
    plot_dsa_bg = plot(0, dsa_bg)
    plot_dsa_pred = plot(1, dsa_pred)

    ## configure some figure settings
    for fig in figs:
        fig.x_range = figs[0].x_range
        fig.y_range = figs[0].y_range
        fig.x_range.range_padding = 0
        fig.y_range.range_padding = 0
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.title.align = 'center'

    def update(f, l):
        ## update the fixed image
        global fixed_id
        if f != fixed_id:
            dsa_bg = scale(test_data[f] - test_data[moving_id]).squeeze()
            plot_dsa_bg.data_source.data['image'] = [convert(dsa_bg)]
            fixed_id = f

        ## predict the registration and update the moved image
        inputs = [test_data[moving_id][None], test_data[f][None], np.array([[l]])]
        predicted = model.predict(inputs)[0].squeeze()
        dsa_pred = scale(test_data[fixed_id].squeeze() - predicted)
        plot_dsa_pred.data_source.data['image'] = [convert(dsa_pred)]

        ## update the images
        bokeh.io.push_notebook()

    ## configure interactive sliders to modify the
    ## input image pair and hyperparameter
    style = {'description_width': '200px'}
    layout = {'width': '600px'}
    f_s = ipywidgets.IntSlider(description='Fixed Image ID',
                              min=0, max=len(test_data) - 1, value=fixed_id,
                              continuous_update=False, layout=layout, style=style)
    l_s = ipywidgets.FloatSlider(description='Hyperparameter λ Value',
                              min=0, max=1, step=1e-2, value=hp0,
                              continuous_update=False, layout=layout, style=style)
    layout = ipywidgets.Layout(display='flex', flex_flow='column', align_items='center')
    ui = ipywidgets.HBox([f_s, l_s], layout=layout)
    out = ipywidgets.interactive_output(update, {'f': f_s, 'l': l_s})
    display(ui, out)  # display slidebars

    ## show bokeh plots
    grid = bokeh.layouts.gridplot([figs],
                                  sizing_mode='scale_width',
                                  toolbar_location='below')

    handle = bokeh.io.show(grid, notebook_handle=True)


def run_interactive_search_2p(model, test_data, hp0=[0.5, 0.5]):
    ## configure initial image pair to align
    moving_id = 0
    global fixed_id
    fixed_id = len(test_data)//2

    ## run the first model prediction, which tends to be slower
    ## since tensorflow needs to initialize some extra elements
    inputs = [test_data[moving_id][None], test_data[fixed_id][None], np.array([[hp0[0], hp0[1]]])]
    predicted = model.predict(inputs)[0].squeeze()

    ## initialize figures for plotting
    figs = [bokeh.plotting.figure(title=t) for t in ['DSA-BG', 'DSA-Pred']]

    ## utilities for plotting image figures
    scale = lambda im: (im - im.min()) / (im.max() - im.min())
    mapper = bokeh.models.LinearColorMapper(palette='Greys256', low=0, high=255)
    convert = lambda im: np.flip(im * 255, 0).astype('uint8')
    plot = lambda id, im: figs[id].image(image=[convert(im)],
                                         x=0, y=0,
                                         dw=im.shape[0], dh=im.shape[1],
                                         color_mapper=mapper)

    ## plot the initial images
    dsa_bg = scale(test_data[fixed_id] - test_data[moving_id]).squeeze()
    dsa_pred = scale(test_data[fixed_id].squeeze() - predicted)
    plot_dsa_bg = plot(0, dsa_bg)
    plot_dsa_pred = plot(1, dsa_pred)

    ## configure some figure settings
    for fig in figs:
        fig.x_range = figs[0].x_range
        fig.y_range = figs[0].y_range
        fig.x_range.range_padding = 0
        fig.y_range.range_padding = 0
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.title.align = 'center'

    def update(f, l, a):
        ## update the fixed image
        global fixed_id
        if f != fixed_id:
            dsa_bg = scale(test_data[f] - test_data[moving_id]).squeeze()
            plot_dsa_bg.data_source.data['image'] = [convert(dsa_bg)]
            fixed_id = f

        ## predict the registration and update the moved image
        inputs = [test_data[moving_id][None], test_data[f][None], np.array([[l, a]])]
        predicted = model.predict(inputs)[0].squeeze()
        dsa_pred = scale(test_data[fixed_id].squeeze() - predicted)
        plot_dsa_pred.data_source.data['image'] = [convert(dsa_pred)]

        ## update the images
        bokeh.io.push_notebook()

    ## configure interactive sliders to modify the
    ## input image pair and hyperparameter
    style = {'description_width': '200px'}
    layout = {'width': '600px'}
    f_s = ipywidgets.IntSlider(description='Fixed Image ID',
                              min=0, max=len(test_data) - 1, value=fixed_id,
                              continuous_update=False, layout=layout, style=style)
    l_s = ipywidgets.FloatSlider(description='Hyperparameter λ Value',
                              min=0, max=1, step=1e-2, value=hp0[0],
                              continuous_update=False, layout=layout, style=style)
    a_s = ipywidgets.FloatSlider(description='Hyperparameter α Value',
                              min=0, max=1, step=1e-2, value=hp0[1],
                              continuous_update=False, layout=layout, style=style)
    layout = ipywidgets.Layout(display='flex', flex_flow='column', align_items='center')
    ui = ipywidgets.HBox([f_s, l_s, a_s], layout=layout)
    out = ipywidgets.interactive_output(update, {'f': f_s, 'l': l_s, 'a': a_s})
    display(ui, out)  # display slidebars

    ## show bokeh plots
    grid = bokeh.layouts.gridplot([figs],
                                  sizing_mode='scale_width',
                                  toolbar_location='below')

    handle = bokeh.io.show(grid, notebook_handle=True)


def scaling(images):
    images = (images - images.min()) / (images.max() - images.min())
    images = images * 4095.
    return images

def matching(image, matchImage):
    """
    Adjust "image" to match "matchImage"
    The shape of both images: (N, H, W), where H = W
    
    ## Example: in my case, I adjust multiple DSA seqences to their mean
    dsa_cases = []
    for model_name in models.keys():
        dsa = dsa_predict(model_name, xseq_bg, xseq_ct)  # predicted dsa
        dsa = scaling(dsa.squeeze())
        dsa_cases.append(dsa)
    
    matchDSA = np.mean(dsa_cases, axis=0)
    for i in range(len(dsa_cases)):
        dsa_cases[i] = matching(image=dsa_cases[i], matchImage=matchDSA)    
    """
    
    ## Take the center of the frame to avoid issues with collumnation
    offs = 0  # Now, no offset is applied
    lb = 0+offs
    ub = image.shape[-1]-offs
    
    firstFrameMean = np.mean(image[0,lb:ub,lb:ub])
    firstFrameMeanMatch = np.mean(matchImage[0,lb:ub,lb:ub]) 
    
    image_centered = image - firstFrameMean
    matchImage_centered = matchImage-firstFrameMeanMatch
    
    ratio_std = np.std(matchImage_centered[:,lb:ub,lb:ub]) / np.std(image_centered[:,lb:ub,lb:ub])
    image_new = image_centered * ratio_std
    image_new = image_new + firstFrameMeanMatch
    
    maximum = np.amax(matchImage)
    minimum = np.amin(matchImage)
    image_new[image_new > maximum] = maximum
    image_new[image_new < minimum] = minimum
    return image_new