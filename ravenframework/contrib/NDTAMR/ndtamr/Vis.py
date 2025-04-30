"""
    The Vis module contains visualation functions for the tree objects.
"""

from __future__ import print_function, division
import numpy as np
from matplotlib.pyplot import subplots


def grid_lines(node,dims=[0,1],slice_=None):
    """
    Output the pair of lines which split the cell.

    Parameters
    ----------
    node : NDTree.Node
        The node of the tree we want to draw

    dims : list
        The two dimensions we will be plotting

    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice_ going through 0.2
        in the last dimension.

    Returns
    -------
    final_lines : list
        The two lines which split the cell

    """
    if node.leaf:
        return None

    dx = 2.**(-node.global_index[0]-1)
    indx = np.array(node.global_index[1:])

    coords = np.array(node.coords)
    ndx = np.array(node.dx)
    if slice_ is not None:
        for s in slice_:
            if not (s[1] >= coords[s[0]] and s[1] < coords[s[0]]+ndx[s[0]]):
                return [None,None]

    i,j = indx[dims]
    idx,jdx = np.array(node.dx)[dims] / 2
    istart,jstart = np.array(node.xmin)[dims]
    i_line = [ (istart + idx*(2*i+1),jstart+jdx*(2*j)),(istart+idx*(2*i+1),jstart+jdx*(2*(j+1)))]
    j_line = [ (istart + idx*(2*i),jstart+jdx*(2*j+1)),(istart+idx*(2*(i+1)),jstart+jdx*(2*j+1))]
    final_lines =  [i_line,j_line]
    return final_lines



def generate_grid(node,dims=[0,1],slice_=None,max_level=np.inf,save=None):
    """
    Returns a list of lines which constitute the grid.

    Parameters
    ----------
    node : NDTree.Node
        The node of the tree we want to draw
    dims : list
        The two dimensions we will be plotting
    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    max_level : int
        The deepest level in the grid we want to display.
        This is useful for showing the progression of the refinement.
    save: str
        If save is not None then save the grid lines to a file with
        filename save

    Returns
    -------
    grid : list
        Final list of grid lines

    """
    lines = []
    node.walk(node_func=lambda x: lines.extend(grid_lines(x,dims=dims,slice_=slice_) if x.global_index[0]<max_level else [None,None]))

    grid = []
    for line in lines:
        if line is not None:
            grid.append( [
                (line[0][0], line[0][1]),
                (line[1][0],line[1][1])])



    if save is not None:
        np.array(grid).tofile(save)

    return grid

def grid_plot(node,dims=[0,1],slice_=None,max_level=np.inf,
             fig=None,ax=None,lw=.5,colors='grey',figsize=(6,6),
             save=None,savefig=None):
    """
    Draws the tree's grid.

    Parameters
    ----------
    node : NDTree.Node
        The node of the tree we want to draw

    dims : list
        The two dimensions we will be plotting

    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    max_level : int
        The deepest level in the grid we want to display.
        This is useful for showing the progression of the refinement.
    fig : matplotlib.figure
        The figure object to plot on
    ax : matplotlib.axis.Axis
        The axis object to plot on
    lw : float
        The linewidth of the gridlines
    colors : str
        The color of the gridlines
    figsize : tuple
        The figure size
    save: str
        If save is not None then save the gridlines to a file with
        filename save
    savefig : str
        If save is not None then save the figure to a file with
        filename savefig

    Returns
    -------
    fig : matplotlib.figure
        The final figure object
    ax : matplotlib.axis.Axis
        The final axis object


    """
    import matplotlib.collections as mc

    first_plot = ax is None
    if first_plot:
        fig,ax = subplots(figsize=figsize)


    xmin = np.array(node.xmin)
    xmax = np.array(node.xmax)

    grid = generate_grid(node,dims=dims,slice_=slice_,max_level=max_level,save=save)
    lc = mc.LineCollection(grid,colors=colors,lw=lw)

    ax.add_collection(lc)




    if first_plot:
        xmin = xmin[dims]
        xmax = xmax[dims]
        ax.set_xlim((xmin[0],xmax[0]))
        ax.set_ylim((xmin[1],xmax[1]))

        ax.minorticks_on()
        ax.set_xlabel('$x_{:d}$'.format(dims[0]+1),fontsize=20)
        ax.set_ylabel('$x_{:d}$'.format(dims[1]+1),fontsize=20)
        ax.tick_params(labelsize=16)
        fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig,bbox_inches='tight')
    return fig,ax
def convert_to_uniform(tree,dims=[0,1],slice_=None,q=None,func=lambda x: x,mask=lambda x: False,alpha_func=lambda x: 1,pad=None):
    """
    Convert the tree to a numpy array for fast (and versitile) plotting.

    Parameters
    ----------
    tree : NDTree.Node
        The tree we want to convert

    dims : list
        The two dimensions we will be plotting

    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    q : str
        The data column to plot from each leaf's Data object
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.
    mask : function
        If we want to mask any values in the final plot we can
        set that criterion through the mask function.
    alpha_func : function
        If we want to change the alpha values in our final plot
        we can set the alpha values through this function
    pad : float
        If not None this will pad the final 2D array with the value
        set by pad.

    Returns
    -------
    result : ndarray
        The final 2D array
    alpha : ndarray
        The final 2D array of alpha values

    """


    if tree.dim > len(dims) and slice_ is None:
        slice_ = [(-1,0)]


    xmin = np.array(tree.xmin)
    xmax = np.array(tree.xmax)

    lmax = tree.depth()

    result = np.zeros((2**lmax,2**lmax))
    alpha = np.zeros((2**lmax,2**lmax)) + 1.
    mask_arr = np.zeros(result.shape).astype(bool)
    leaves = tree.list_leaves(attr='self')
    for n in leaves:
        lvl = n.global_index[0]
        indices = np.array(n.global_index[1:])
        dx = np.array(n.dx)
        coords = np.array(n.coords)
        i,j = indices[dims]

        if slice_ is None:
            if q is None:
                d = func(n)
            else:
                d = func(getattr(n.data,q))
            m = mask(n)
            a = alpha_func(n)
            if lvl == lmax:
                result[i,j] = d
                alpha[i,j] = a
                mask_arr[i,j] = m
            else:
                fac = 2**(lmax-lvl)
                result[fac*i:fac*(i+1),fac*j:fac*(j+1)] = d
                alpha[fac*i:fac*(i+1),fac*j:fac*(j+1)] = a
                mask_arr[fac*i:fac*(i+1),fac*j:fac*(j+1)] = m
        else:
            good = all([min(max(xmin[s[0]],s[1]),xmax[s[0]]) >= coords[s[0]] and min(max(xmin[s[0]],s[1]),xmax[s[0]]) <= coords[s[0]]+dx[s[0]] for s in slice_])
            if good:
                if q is None:
                    d = func(n)
                else:
                    d = func(getattr(n.data,q))
                m = mask(n)
                a = alpha_func(n)
                if lvl == lmax:
                    result[i,j] = d
                    alpha[i,j] = a
                    mask_arr[i,j] = m
                else:
                    fac = 2**(lmax-lvl)
                    result[fac*i:fac*(i+1),fac*j:fac*(j+1)] = d
                    alpha[fac*i:fac*(i+1),fac*j:fac*(j+1)] = a
                    mask_arr[fac*i:fac*(i+1),fac*j:fac*(j+1)] = m

    if pad is not None:
        temp = np.vstack((np.zeros((2**lmax,))+pad,result,np.zeros((2**lmax,))+pad))
        result = np.hstack((np.zeros((2**lmax+2,1))+pad,temp,np.zeros((2**lmax+2,1))+pad))

        temp = np.vstack((np.zeros((2**lmax,)),alpha,np.zeros((2**lmax,))))
        alpha = np.hstack((np.zeros((2**lmax+2,1)),temp,np.zeros((2**lmax+2,1))))

        temp = np.vstack((np.zeros((2**lmax,)),mask_arr,np.zeros((2**lmax,))))
        mask_arr = np.hstack((np.zeros((2**lmax+2,1)),temp,np.zeros((2**lmax+2,1))))

    result = np.ma.masked_array(result,mask_arr)
    alpha = np.ma.masked_array(alpha,mask_arr)
    return result,alpha

def convert_to_uniform_integrate(tree,dims=[0,1],dim=-1,take_min=False,take_max=False,slice_=None,q=None,func=lambda x: x,mask=lambda x: False,alpha_func=lambda x: 1.,pad=None):
    """
    Convert the tree to a numpy array for fast (and versitile) plotting.
    This additionally integrates the tree across the dimension given

    Parameters
    ----------
    tree : NDTree.Node
        The tree we want to convert

    dims : list
        The two dimensions we will be plotting
    dim : int
        The dimension to integrate over.
    take_min : bool
        If True then take the minimum value along the
        dimension set by dim.
    take_max : bool
        If True then take the maximum value along the
        dimension set by dim.


    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    q : str
        The data column to plot from each leaf's Data object
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.
    mask : function
        If we want to mask any values in the final plot we can
        set that criterion through the mask function.
    alpha_func : function
        If we want to change the alpha values in our final plot
        we can set the alpha values through this function
    pad : float
        If not None this will pad the final 2D array with the value
        set by pad.

    Returns
    -------
    result : ndarray
        The final 2D array
    alpha : ndarray
        The final 2D array of alpha values

    """
    xmin = [i for i in tree.xmin]
    xmax = [i for i in tree.xmax]

    xi = xmin.pop(dim)
    xo = xmax.pop(dim)

    lmax = tree.depth()

    tot = 2**(-lmax)

    result = np.zeros((2**lmax,2**lmax))
    norm = np.zeros((2**lmax,2**lmax))
    if take_min:
        result += 1e99
    elif take_max:
        result -= 1e99
    alpha = np.zeros((2**lmax,2**lmax))
    mask_arr = np.zeros(result.shape)
    leaves = tree.list_leaves(attr='self')

    for n in leaves:
        lvl = n.global_index[0]
        indices = np.array(n.global_index[1:])
        dx = np.array(n.dx)
        coords = np.array(n.coords)
        weight = 2.**(-lvl)
        i,j = indices[dims]
        if q is None:
            d = func(n)
        else:
            d = getattr(n.data,q)
        m = mask(n)
        a = alpha_func(n)
        if ~(np.isnan(d)|np.isinf(d)):
            try:
                if lvl == lmax:
                    if m < 1:
                        if take_min:
                            result[i,j] = min(d,result[i,j])
                        else:
                            result[i,j] += weight*d
                        norm[i,j] += weight
                    alpha[i,j] += weight*a
                    mask_arr[i,j] += weight*m
                else:
                    fac = 2**(lmax-lvl)

                    if m < 1:
                        if take_min:
                            result[fac*i:fac*(i+1),fac*j:fac*(j+1)] = np.minimum(result[fac*i:fac*(i+1),fac*j:fac*(j+1)],d)
                        elif take_max:
                            result[fac*i:fac*(i+1),fac*j:fac*(j+1)] = np.maximum(result[fac*i:fac*(i+1),fac*j:fac*(j+1)],d)
                        else:
                            result[fac*i:fac*(i+1),fac*j:fac*(j+1)] += weight*d
                        norm[fac*i:fac*(i+1),fac*j:fac*(j+1)] += weight
                    mask_arr[fac*i:fac*(i+1),fac*j:fac*(j+1)] += weight*m
                    alpha[fac*i:fac*(i+1),fac*j:fac*(j+1)] += weight*a
            except TypeError:
                print(i,j,d,weight,m)
    if q is not None:
        result = func(result)
    if pad is not None:
        temp = np.vstack((np.zeros((2**lmax,))+pad,result,np.zeros((2**lmax,))+pad))
        result = np.hstack((np.zeros((2**lmax+2,1))+pad,temp,np.zeros((2**lmax+2,1))+pad))

        temp = np.vstack((np.zeros((2**lmax,)),alpha,np.zeros((2**lmax,))))
        alpha = np.hstack((np.zeros((2**lmax+2,1)),temp,np.zeros((2**lmax+2,1))))

        temp = np.vstack((np.zeros((2**lmax,)),mask_arr,np.zeros((2**lmax,))))
        mask_arr = np.hstack((np.zeros((2**lmax+2,1)),temp,np.zeros((2**lmax+2,1))))
    result = np.ma.masked_array(result,mask_arr==1)
    return result,alpha
def _test_slice(n,slice_):
        """
        tests if the given node satisfies the slice condition.
        returns none if it does not.
        """
        if slice_ is None:
            return True
        lvl = n.global_index[0]
        indices = np.array(n.global_index[1:])
        dx = np.array(n.dx)
        coords = np.array(n.coords)


        return all([s[1] >= coords[s[0]] and s[1] < coords[s[0]]+dx[s[0]] for s in slice_])
def _get_slice(tree,dim,q,func,slice_):
    """
    Get the data values satisfying the slice_ condition

    Parameters
    ----------
    tree : NDTree.Node
        The node of the tree we want to draw

    dims : list
        The two dimensions we will be plotting

    q : str
        The data column to plot from each leaf's Data object
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.

    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.


    Returns
    -------
    final_list : list
        List of (coordinate,value) pairs which satisfy the slice_ conditions.

    """
    if tree.dim > 1 and slice_ is None:
        slice_ = [(-1,0)]
    def _lfunc(n,dim,slice_,q,func):
        """
        tests if the given node satisfies the slice condition.
        returns none if it does not.
        """
        lvl = n.global_index[0]
        indices = np.array(n.global_index[1:])
        dx = np.array(n.dx)
        coords = np.array(n.coords)

        good = all([s[1] >= coords[s[0]] and s[1] < coords[s[0]]+dx[s[0]] for s in slice_])
        if good:
            return coords[dim],func(getattr(n.data,q)),n.rflag
        return None,None
    vals = []
    tree.walk(leaf_func = lambda x: vals.append(_lfunc(x,dim,slice_,q,func)))
    final_list = list(filter(lambda x: not None in x,vals))
    return final_list

def line_plot(tree,dim=0,slice_=None,grid=False,rflag=False,q='value',func=lambda x: x,figsize=(8,6),
              fig=None,ax=None,savefig=None,**kargs):
    """
    A 1D line plot for the tree.

    Parameters
    ----------
    tree : NDTree.Node
        The tree we want to draw
    dims : list
        The one dimension we will be plotting
    slice_ : list of tuples
        If the node has more than one dimension,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    q : str
        The data column to plot from each leaf's Data object
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.
    figsize : tuple
        The figure size
    fig : matplotlib.figure
        The figure object to plot on
    ax : matplotlib.axis.Axis
        The axis object to plot on
    savefig : str
        If save is not None then save the figure to a file with
        filename savefig
    **kargs :
        Keyword arguments passed to plt.plot


    Returns
    -------
    fig : matplotlib.figure
        The final figure object
    ax : matplotlib.axis.Axis
        The final axis object

    """
    if ax is None:
        fig,ax = subplots(figsize=(8,6))

    if tree.dim == 1:
        vals=[]
        tree.walk(leaf_func = lambda x: vals.append([x.coords[0], func(getattr(x.data,q)),int(x.rflag)]))
        vals = np.array(vals)
    else:
        vals = np.array(_get_slice(tree,dim,q,func,slice_))


    if grid or rflag:
        ls = kargs.pop('marker','.')
        kargs['marker'] = ls

    if rflag:
        ind = vals[:,2].astype(bool)
        ax.plot(vals[:,0],vals[:,1],**kargs)
        ax.plot(vals[ind,0],vals[ind,1],'.r')
    else:
        ax.plot(vals[:,0],vals[:,1],**kargs)

    ax.set_xlabel('$x$')
    ax.set_ylabel(q)
    ax.minorticks_on()
    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig,bbox_inches='tight')
    return fig,ax

def plot(tree,dims=[0,1],integrate=None,take_min=False,take_max=False,slice_=None,
        q='value',cmap='viridis',rflag=False,func=lambda x: x,mask=lambda x: False,
         alpha_func=lambda x: 1,pad=None,grid=False,figsize=(6,6),fig=None,ax=None,
         labels={},lognorm=False,colorbar=True,cb_kargs={},savefig=None,**kargs):
    """
    A 2D color plot for the tree.

    Parameters
    ----------
    tree : NDTree.Node
        The tree we want to draw

    dims : list
        The two dimensions we will be plotting
    integrate : int
        If not None, then integrate the tree along the
        dimension given by integrate.
    take_min : bool
        If True then take the minimum value along the
        dimension set by integrate.
    take_max : bool
        If True then take the maximum value along the
        dimension set by integrate.
    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    q : str
        The data column to plot from each leaf's Data object
    cmap : str
        The colormap to use
    rflag : bool
        If True indicate which cells are flagged for refinement.
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.
    mask : function
        If we want to mask any values in the final plot we can
        set that criterion through the mask function.
    alpha_func : function
        If we want to change the alpha values in our final plot
        we can set the alpha values through this function
    grid : bool
        If True then we additionally plot the grid lines.
    colors : str
        The color of the grid lines
    figsize : tuple
        The figure size
    fig : matplotlib.figure
        The figure object to plot on
    ax : matplotlib.axis.Axis
        The axis object to plot on
    labels : dict
        The axis labels. For example, {'x':'x', 'y':'y'}
    lognorm : bool
        If True the colorbar will be log scale.
    colorbar : bool
        If False then do not show the colorbar.
    cb_kargs = dict
        Keyword arguments passed to the _create_colorbar() function.
    **kargs :
        Keyword arguments passed to plt.imshow
    savefig : str
        If save is not None then save the figure to a file with
        filename savefig


    Returns
    -------
    fig : matplotlib.figure
        The final figure object
    ax : matplotlib.axis.Axis
        The final axis object

    """
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    if ax is None:
        fig,ax = subplots(figsize=figsize)


    xmin = np.array(tree.xmin)
    xmax = np.array(tree.xmax)
    xmin1 = xmin[dims]
    xmax1 = xmax[dims]

    if pad is not None:
        lmax = tree.depth()
        dx0 = (xmax1[0]-xmin1[0])/2**lmax
        dx1 = (xmax1[1]-xmin1[1])/2**lmax
    else:
        dx0 = 0
        dx1 = 0

    if integrate is not None:
        res,alpha = convert_to_uniform_integrate(tree,dim=integrate,take_min=take_min,take_max=take_max,dims=dims,slice_=slice_,q=q,func=func,mask=mask,alpha_func=alpha_func,pad=pad)
    else:
        res,alpha = convert_to_uniform(tree,dims=dims,slice_=slice_,q=q,func=func,mask=mask,alpha_func=alpha_func,pad=pad)
    origin = kargs.pop('origin','lower')
    interpolation = kargs.pop('interpolation','none')
    vmin = kargs.pop('vmin',res.min())
    vmax = kargs.pop('vmax',res.max())




    if lognorm:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin,vmax=vmax)
    c = cm.ScalarMappable(norm=norm,cmap=cmap)
    cv = c.to_rgba(res.data.T)
    cv[:,:,-1] = alpha.T


    ax.imshow(cv,extent=(xmin1[0]-dx0,xmax1[0]+dx0,xmin1[1]-dx1,xmax1[1]+dx1),origin=origin,interpolation=interpolation,**kargs)

    cb = None
    if colorbar:
        cb = _create_colorbar(ax,vmin=vmin,vmax=vmax,cmap=cmap,**cb_kargs)

    if rflag:
        coords = []
        cfunc = lambda x: coords.append(x.get_coords(shift=True) if x.rflag and _test_slice(x,slice_) else None)
        tree.walk(leaf_func=cfunc)
        for c in coords:
            if c is not None:
                ax.plot(c[dims[0]],c[dims[1]],'r.')





    ax.set_xlim((xmin1[0],xmax1[0]))
    ax.set_ylim((xmin1[1],xmax1[1]))


    if grid:
        grid_plot(tree,dims=dims,slice_=slice_,fig=fig,ax=ax,**kargs)
    ax.minorticks_on()
    fontsize = labels.pop('fontsize',20)
    xlbl = labels.pop('x','$x_{:d}$'.format(dims[0]+1))
    ylbl = labels.pop('y','$x_{:d}$'.format(dims[1]+1))
    ax.set_xlabel(xlbl,fontsize=fontsize)
    ax.set_ylabel(ylbl,fontsize=fontsize)

    ax.tick_params(labelsize=fontsize)
    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig,bbox_inches='tight')
    return fig,ax,cb,res.T,alpha.T
def contour(tree,dims=[0,1],integrate=None,take_min=False,take_max=False,slice_=None,q='value',
        labels={},rflag=False,func=lambda x: x,mask=lambda x: False,alpha_func=lambda x: 1.,
        pad=None,grid=False,colorbar=True,cb_kargs={},figsize=(6,6),fig=None,ax=None,savefig=None,**kargs):
    """
    Draw a contour plot for the tree.

    Parameters
    ----------
    tree : NDTree.Node
        The tree we want to draw
    Nconts : int
        The number of contours to draw
    dims : list
        The two dimensions we will be plotting
    integrate : int
        If not None, then integrate the tree along the
        dimension given by integrate.
    take_min : bool
        If True then take the minimum value along the
        dimension set by integrate.
    take_max : bool
        If True then take the maximum value along the
        dimension set by integrate.
    slice_ : list of tuples
        If the node has more than two dimensions,
        slice_ indicates the values for the extra
        dimensions.
        For example, slice_=[(-1,0.2)] indicates
        that we want the slice to go through 0.2
        in the last dimension.
    q : str
        The data column to plot from each leaf's Data object
    cmap : str
        The colormap to use
    rflag : bool
        If True indicate which cells are flagged for refinement.
    func : function
        Before plotting the data we pass it through this function.
        By default this just returns the value passed to it.
    mask : function
        If we want to mask any values in the final plot we can
        set that criterion through the mask function.
    alpha_func : function
        If we want to change the alpha values in our final plot
        we can set the alpha values through this function
    pad : float
        If not None this will pad the final 2D array with the value
        set by pad.
    grid : bool
        If True then we additionally plot the grid lines.
    colorbar : bool
        If False then do not show the colorbar.
    cb_kargs = dict
        Keyword arguments passed to the _create_colorbar() function.
    figsize : tuple
        The figure size
    fig : matplotlib.figure
        The figure object to plot on
    ax : matplotlib.axis.Axis
        The axis object to plot on
    savefig : str
        If save is not None then save the figure to a file with
        filename savefig
    **kargs :
        Keyword arguments passed to plt.contour


    Returns
    -------
    fig : matplotlib.figure
        The final figure object
    ax : matplotlib.axis.Axis
        The final axis object

    """
    if ax is None:
        fig,ax = subplots(figsize=figsize)


    xmin = np.array(tree.xmin)
    xmax = np.array(tree.xmax)
    xmin1 = xmin[dims]
    xmax1 = xmax[dims]

    if pad is not None:
        lmax = tree.depth()
        dx0 = (xmax1[0]-xmin1[0])/2**lmax
        dx1 = (xmax1[1]-xmin1[1])/2**lmax
    else:
        dx0 = 0
        dx1 = 0

    if integrate is not None:
        res,alpha = convert_to_uniform_integrate(tree,dim=integrate,take_min=take_min,take_max=take_max,dims=dims,slice_=slice_,q=q,func=lambda x: x,mask=lambda x: False,alpha_func=alpha_func,pad=pad)
        res = func(res)
    else:
        res,alpha = convert_to_uniform(tree,dims=dims,slice_=slice_,q=q,func=func,mask=lambda x: False,alpha_func=alpha_func,pad=pad)

    origin = kargs.pop('origin','lower')

    ax.contour(res.T,extent=(xmin1[0]-dx0,xmax1[0]+dx0,xmin1[1]-dx1,xmax1[1]+dx1),origin=origin,**kargs)


    vmin = kargs.pop('vmin',res.min())
    vmax = kargs.pop('vmax',res.max())
    cmap = kargs.pop('cmap','viridis')


    cb = None
    if colorbar:
        cb = _create_colorbar(ax,vmin=vmin,vmax=vmax,cmap=cmap,**cb_kargs)






    ax.set_xlim((xmin1[0],xmax1[0]))
    ax.set_ylim((xmin1[1],xmax1[1]))

    ax.minorticks_on()
    fontsize = labels.pop('fontsize',20)
    xlbl = labels.pop('x','$x_{:d}$'.format(dims[0]+1))
    ylbl = labels.pop('y','$x_{:d}$'.format(dims[1]+1))
    ax.set_xlabel(xlbl,fontsize=fontsize)
    ax.set_ylabel(ylbl,fontsize=fontsize)

    ax.tick_params(labelsize=16)

    if grid:
        grid_plot(tree,dims=dims,colors='grey',slice_=slice_,fig=fig,ax=ax,**kargs)
    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig,bbox_inches='tight')
    return fig,ax
def _create_colorbar(ax,vmin,vmax,cax=None,log=False,cmap='viridis',**kargs):
    """
    Function to create a colorbar at the top of a plot

    Parameters
    ----------
    ax : matplotlib.axis.Axis
        The axis object we want to draw the colorbar over.

    vmin : float
        The minimum value for the colorbar

    vmax : float
        The maximum value for the colorbar

    log : bool
        If log==True then the colorscale is log scale
    cmap : str
        The colormap to use for the colorbar
    **kargs :
        Extra keyword arguments which are passed to matplotlib.colorbar.ColorbarBase


    Returns
    -------
    cb : matplotlib.colorbar
        The final colorbar.

    """
    import matplotlib
    import matplotlib.cm
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    labelsize = kargs.pop('labelsize',12)
    degrees = kargs.pop('degrees',False)
    upper_lim = kargs.pop('upper_lim',False)
    lower_lim = kargs.pop('lower_lim',False)

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top',size='3%',pad=.05)


    if log:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.get_cmap(cmap)
    cb = matplotlib.colorbar.ColorbarBase(ax=cax,cmap=cmap,norm=norm,orientation='horizontal',**kargs)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=labelsize)



    if degrees:
        cb_lbls = cb.ax.get_xticklabels()
        for l in cb_lbls:
            l.set_text('${}^\\circ$'.format(l.get_text()))
        cb.ax.set_xticklabels(cb_lbls)
    if upper_lim:
        cb_lbls = cb.ax.get_xticklabels()
        cb_lbls[-1].set_text('$>$'+cb_lbls[-1].get_text())
        cb.ax.set_xticklabels(cb_lbls)
    if lower_lim:
        cb_lbls = cb.ax.get_xticklabels()
        cb_lbls[0].set_text('$<$'+cb_lbls[0].get_text())
        cb.ax.set_xticklabels(cb_lbls)


    return cb
