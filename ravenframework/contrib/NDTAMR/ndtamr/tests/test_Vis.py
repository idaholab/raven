#import matplotlib
#matplotlib.use('Agg')
#from matplotlib.testing.decorators import image_comparison
#import matplotlib.pyplot as plt
#import numpy as np
#from ..Vis import *
#from .text_AMR import make_example_tree2, make_example_tree_uniform
#
#@image_comparison(baseline_images=['plotex'],extensions=['png'])
#def test_grid_plot():
#    fig,ax=plt.subplots()
#    x = np.linspace(-np.pi,np.pi,1000)
#    y = np.sin(x)
#    ax.plot(x,y)
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('$y$')
#    ax.minorticks_on()
#    fig.tight_layout()
#    return
        
#    def test__create_colorbar(self):
#
#    def test__get_slice(self):
#
#    def test_contour(self):
#
#    def test_convert_to_uniform(self):
#
#    def test_generate_grid(self):
#
#    def test_grid_lines(self):
#
#    def test_grid_plot(self):
#
#    def test_line_plot(self):
#
#    def test_plot(self):
#
#    def test_subplots(self):