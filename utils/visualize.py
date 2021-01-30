import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from matplotlib import patches
from matplotlib.collections import PatchCollection


def plot_bounding_box_patch(pred_nms, freq_scale, start_time):
    patch_collect = []
    for bb in range(len(pred_nms)):
        xx = pred_nms[bb][0]  #- start_time
        ww = pred_nms[bb][2]
        yy = (pred_nms[bb][1]) #/ freq_scale
        hh = (pred_nms[bb][3]) #/ freq_scale
        patch_collect.append(patches.Rectangle((xx,yy),ww,hh, linewidth=1, edgecolor='w',
                                 facecolor='none', alpha=1.0))
    return patch_collect

def plot_bounding_box_predictions(pred_nms):
    patch_collect = []
    for bb in range(len(pred_nms)):
        xx = pred_nms[bb][0]  #- start_time
        ww = pred_nms[bb][2]
        yy = (pred_nms[bb][1]) #/ freq_scale
        hh = (pred_nms[bb][3]) #/ freq_scale
        patch_collect.append(patches.Rectangle((xx,yy),ww,hh, linewidth=1, edgecolor='r',
                                 facecolor='none', alpha=1.0))
    return patch_collect


def create_box_image(spec, fig, detections_ip, predictions, start_time, end_time, duration, params, max_val, hide_axis=True):
    # filter detections
    detections = []
    for bb_anns in detections_ip:
        if ((bb_anns[0]) >= 0) and ((bb_anns[0] + bb_anns[2]) < spec.shape[1]):
            detections.append(bb_anns)

    p = []
    for bb_anns in predictions:
        if ((bb_anns[0]) >= 0) and ((bb_anns[0] + bb_anns[2]) < spec.shape[1]):
            p.append(bb_anns)
    # create figure
    freq_scale = 1000  # turn Hz in kHz
    y_extent = [0, spec.shape[1], 0, spec.shape[0]]#[0, duration, params['min_freq']//freq_scale, params['max_freq']//freq_scale]

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    if hide_axis:
        ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(spec, aspect='auto', cmap='plasma', extent=y_extent, vmin=0, vmax=max_val)
    boxes = plot_bounding_box_patch(detections, freq_scale, start_time)
    pred_boxes = plot_bounding_box_predictions(p)
    ax.add_collection(PatchCollection(boxes, match_original=True))
    ax.add_collection(PatchCollection(pred_boxes, match_original=True))
    plt.grid(False)

