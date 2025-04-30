"""
    The AMR module contains functions which adaptively refine the domain.
"""
from __future__ import print_function, division
import numpy as np
from .Vis import plot,line_plot
import matplotlib.pyplot as plt

def compression(tree):
    """
    Print some statistics about the efficiency of the AMR grid.
    """
    depth = tree.depth()
    nleaves = len(tree.list_leaves())

    tot = (2**depth)**tree.dim

    print('{:d} points out of {:d}^{:d} = {:d} for full grid'.format(nleaves,2**depth,tree.dim,tot))
    print('You have saved a factor of {:.2f}'.format(tot/nleaves))
    print('With a compression factor of {:.2f}%'.format((1-nleaves/tot)*100))

def clear_refine(tree):
    """
    Set all refinemnet flags to False
    """
    tree.walk(leaf_func=lambda x: setattr(x,'rflag',False))

def start_refine(tree):
    """
    Look through leaves and split if flagged for refinement.
    """

    def _do_split(node,count):
        """
        Split the node if it is flagged for refinement.

        Parameters
        ----------
        node : NDtree.Node
            Node we are evaluating
        count : list
            The list of nodes which have been refined
        """
        if node.rflag:
            node.rflag = False
            count.append(node.name)
            node.split()

    total = []
    tree.walk(leaf_func = lambda x: _do_split(x,total))

    return len(total)

def start_derefine(tree):
    """
    Look through leaves and derefine if needed.
    """

    def _do_unsplit(x,count):
        """
        Unsplit the node if it is flagged for derefinement.

        Parameters
        ----------
        node : NDtree.Node
            Node we are evaluating
        count : list
            The list of nodes which have been derefined
        """
        if x.rflag and x is not None:
            count.append(x.name)
            x.parent.unsplit()
            x.rflag = False
    total = []
    tree.walk(leaf_func = lambda x: _do_unsplit(x,total))
    return len(total)
def refine(tree,tol=.2,eps=.01,finish=True,show=False,extent=2,plot_kargs={},**kargs):

    """
    The main AMR routine which evaluates and refines
    each node in the tree.

    Parameters
    ----------
    tree : NDTree.node
        The tree we want to refine
    tol : float
        The tolerance level for refinement
    eps : float
        Helps with smoothing out flucuations in the
        refinement variable
    show : bool
        If True plot the domain showing which cells
        will be refined
    **kargs :
        Keyword arguments passed to the refinement_check function

    Returns
    -------
    total : int
        The total number of refined cells

    """
    depth = tree.depth()
    values = []
    for lvl in range(depth+1)[::-1]:
        tree.walk(target_level=lvl,
                  leaf_func = lambda x: values.append(refinement_check(x,tol=tol,eps=eps,extent=extent,**kargs)[1]))
    for lvl in range(depth+1)[::-1]:
        tree.walk(target_level=lvl,leaf_func = lambda x: neighbor_check(x,extent=extent))
    if show:
        print('Minimum:', min(values))
        print('Maximum:',max(values))
        print('Median:',np.median(values))
        print('Average:',np.mean(values))
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        axes[0].hist(values,bins=20,histtype='step',lw=3,color='k')
        axes[1].hist(values,bins=20,histtype='step',cumulative=True,density=True,lw=3,color='k')


        for ax in axes:
            ax.set_xlim(0,1)
            ax.set_xlabel('$\\epsilon$',fontsize=20)
            ax.axvline(tol,c='k',ls='--')
            ax.tick_params(labelsize=20)
            ax.minorticks_on()
        if tree.dim == 1:
            line_plot(tree,rflag=True,**plot_kargs)
        else:
            plot(tree,grid=True,rflag=True,**plot_kargs)
    total = 0
    if finish:
        total = start_refine(tree)
    return total


def neighbor_check(node,**kargs):
    """
    Check that if a finer neighbor refined we also refine.
    This enforces that the maximum discrepency in neighbor levels
    is one.
    """
    if not node.rflag:
        return
    _,_,_,neighbors = node.find_neighbors(**kargs)

    for n in neighbors:
        if n is not None:
            if n.leaf:
                n.rflag = True

def refinement_flash(leaf,nodes,tol=.2,eps=0.01,min_value=1e-5,reverse=False):
    """
    The refinement criteria of L\"{o}hner (1987) modified slightly as in the Flash code.
    This function does not evaulate neighbors which are on a finer level, as
    they should have already been evaulated.

    Parameters
    ----------
    leaf : NDTree.node
        The leaf node we are evaluating
    nodes : list
        List of neighbor leaves from get_refinement_neighbors() function
    tol : float
        The tolerance level for refinement
    eps : float
        Helps with smoothing out flucuations in the
        refinement variable
    min_value : float
        Minimum value for the denominator
    reverse : bool
        If True then we flag the cell if it does not satisfy the
        refinement criteria
    Returns
    -------
    res: bool
        If True we (de)refine this cell.
    value: float
        The numerical value for the refinement criteria

    """

    total_neighbors = len(nodes)
    dim = leaf.dim
    # Get the extent of the stencil using total = (2*ext+1)^dim
    ext = int( (total_neighbors**(1./dim)-1)*.5)
    stride = 2*ext+1
    u = np.zeros((total_neighbors,))
    for i,n in enumerate(nodes):
        if n is None:
                u[i] = 0
        else:
            if n.data is not None:
                u[i] = n.data.get_refinement_data()
            else:
                try:
                    u[i] = n.restrict().get_refinement_data()
                except AttributeError:
                    u[i] = 0
    au = abs(u)

    num = 0
    den = 0
    ifunc = lambda x: sum([l*stride**(dim-1-k) for k,l in enumerate(x)])
    for i in range(dim):
        for j in range(dim):
            if i==j:
                iL = [ext]*dim
                iR = [ext]*dim
                iC = [ext]*dim
                iL[i] -= ext
                iR[i] += ext
                num += (u[ifunc(iR)] - 2*u[ifunc(iC)]+u[ifunc(iL)])**2
                dfac = (abs(u[ifunc(iR)]-u[ifunc(iC)]) + abs(u[ifunc(iL)]-u[ifunc(iC)]))
                dfac += eps*(au[ifunc(iR)] + 2*au[ifunc(iC)]+au[ifunc(iL)])
            else:
                iLL = [ext]*dim
                iRR = [ext]*dim
                iLR = [ext]*dim
                iRL = [ext]*dim
                iC = [ext]*dim
                iLL[i] -= 1
                iLL[j] -= 1
                iRR[i] += 1
                iRR[j] += 1
                iRL[i] += 1
                iRL[j] -= 1
                iLR[i] -= 1
                iLR[j] += 1
                num += (u[ifunc(iRR)]+u[ifunc(iLL)]-u[ifunc(iLR)]-u[ifunc(iRL)])**2
                dfac = eps*(au[ifunc(iRR)]+au[ifunc(iLL)]+au[ifunc(iLR)]+au[ifunc(iRL)])
                dfac += abs(u[ifunc(iRR)]-u[ifunc(iLR)]) + abs(u[ifunc(iLL)]-u[ifunc(iRL)])
            den += dfac**2
    value = np.sqrt(num/max(den,min_value))

    if reverse:
        res = value <= tol
    else:
        res = value > tol
    return res,value

def get_refinement_neighbors(leaf,extent=2):
    """
    Get the list of neighbors used for refinement.
    This combines the neighbor and upper_neighbor list into
    one final list of neighbors

    Parameters
    ----------
    leaf : NDTree.node
        The leaf node we are evaluating

    extent : int
        The extent of the stencil, -extent,...,0,...,extent

    Returns
    -------
    final_list : list
        The final list of neighbors

    """
    offsets, neighbor_indices,neighbors, upper_neighbors = leaf.find_neighbors(extent=extent)
    total_neighbors = len(neighbors)

    # Even if already tagged, still need to check new neighbors
    final_list = [None]*total_neighbors

    for i in range(total_neighbors):

        if upper_neighbors[i] is not None:
            node = upper_neighbors[i]
            if not node.leaf:
                node = neighbors[i]
            final_list[i] = node
    return final_list

def refinement_check(leaf,criteria=refinement_flash,extent=2,**kargs):
    """
    Deterimine neighbors and see if this node should be refined.
    If the node satisfies the criteria, then we also flag all of its
    leaf neighbors.

    Parameters
    ----------
    leaf : NDTree.node
        The leaf node we are evaluating

    criteria : function
        The function which evaluates the refinement criteria.
    extent : int
        The extent of the stencil, -extent,...,0,...,extent
    **kargs :
        Keyword arguments which are passed to the criteria function.

    Returns
    -------
    res: bool
        If True we refine this cell.
    value: float
        The numerical value for the refinement criteria

    """

    # Get neighbors

    final_list = get_refinement_neighbors(leaf,extent=extent)
    res,value = criteria(leaf,final_list,**kargs)

    leaf.rflag = res

#    for node in final_list:
#        if node is not None:
#            if node.leaf:
#                node.rflag  |= res


    return res,value
