from __init__ import *
from scipy import spatial
from collections import namedtuple
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import pyplot as plt
import numpy as np
import os
import PIL
import sys

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
# from points_GUI import *

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

colors = [
    'tab:red',
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:purple',
    'tab:cyan',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive'
]


def print_progress(part, whole):
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()

class tracker_window():

    def __init__(self, dirname="./", fn='filter.npy'):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.fn = os.path.join(self.dirname, fn)
        if os.path.isfile(self.fn):
            self.limits = np.load(self.fn)
            self.limit_lower, self.limit_upper = self.limits.min(), self.limits.max()
        self.data_changed = False
        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        # start with vmin and vmax at extremes
        self.vmin = 0
        self.vmax = np.iinfo(self.image.dtype).max
        self.vmax_possible = np.copy(self.vmax)
        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        # fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        fig_left, fig_bottom, fig_width, fig_height = .1, .1, .75, .8
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider controlling frames
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # the vmin slider
        vminframe = plt.axes([fig_left + fig_width + .02, 0.1, .02, .05 + .7])
        self.vmin = Slider(
            vminframe, 'min', 0, self.vmax_possible,
            valinit=0, valfmt='%d', color='k', orientation='vertical')
        self.vmin.on_changed(self.show_image)
        # the vmax slider
        vmaxframe = plt.axes([fig_left + fig_width + .1, 0.1, .02, .05 + .7])
        self.vmax = Slider(
            vmaxframe, 'max', 0, self.vmax_possible, valinit=self.vmax_possible,
            valfmt='%d', color='k', orientation='vertical')
        self.vmax.on_changed(self.show_image)
        # limit both sliders
        self.vmin.slidermax = self.vmax
        self.vmax.slidermin = self.vmin
        # the colorbar in between
        self.cbar_ax = plt.axes([fig_left + fig_width + .06, 0.1, .02, .05 + .7])
        self.colorvals = np.arange(self.vmax_possible)
        self.cbar = self.cbar_ax.pcolormesh([0, 10],
                                            self.colorvals,
                                            self.colorvals[:, np.newaxis],
                                            cmap='viridis', vmin=0, vmax=self.vmax_possible)
        self.cbar_ax.set_xticks([])
        self.cbar_ax.set_yticks([])
        # connect some keys
        # self.cidk = self.figure.canvas.mpl_connect(
        #     'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

    def load_filenames(self):
        ls = os.listdir(self.dirname)
        self.filenames = [os.path.join(self.dirname, f) for f in ls if f.lower().endswith(
            ('.png', '.jpg', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'))]
        self.filenames.sort()

    def load_image(self):
        print(self.curr_frame_index)
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plotthe image
        self.im = np.copy(self.image)
        colorvals = np.copy(self.colorvals)
        # remove values > vmax
        self.im[self.im > self.vmax.val] = 0
        self.figure.axes[0].get_images()[0].set_clim([self.vmin.val, self.vmax.val])
        self.figure.axes[0].get_images()[0].set_data(self.im)
        colorvals[colorvals > self.vmax.val] = 0
        self.cbar.set_array(colorvals)
        self.cbar.set_clim([self.vmin.val, self.vmax.val])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        self.show_image()
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)
        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.change_frame(0)

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)


if __name__ == '__main__':
    folder_UI = folderSelector(
        title='Choose the folder containing the original image stack: ')
    folder_UI.close()
    home_folder = folder_UI.folder
    tracker = tracker_window(home_folder)
    plt.show()
