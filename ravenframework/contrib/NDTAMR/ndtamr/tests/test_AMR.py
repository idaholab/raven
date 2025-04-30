#import ..AMR as amr
from ..AMR import *
from ..NDTree import *
from ..Data import *

def make_example_tree(with_data=False):
    if with_data:
        t = Node(dim=2,data_class=SimpleTest2D,prolongate_func=prolongate_datafunc,
                restrict_func=restrict_datafunc)
    else:
        t = Node(dim=2)
    t.split()
    for c in t.child:
        c.split()
    t.child[0].child[0].split()
    t.child[0].child[3].split()
    t.child[1].child[0].split()
    t.child[1].child[1].split()
    t.child[3].child[0].split()
    return t
def make_example_tree2(with_data=False):
    if with_data:
        t = Node(dim=2,data_class=SimpleTest2D,prolongate_func=prolongate_datafunc,
                restrict_func=restrict_datafunc)
    else:
        t = Node(dim=2)
    t.split()
    for c in t.child:
        c.split()
    t.child[0].child[0].split()
    t.child[0].child[3].split()
    t.child[1].child[0].split()
    t.child[1].child[1].split()
    t.child[3].child[0].split()
    t.child[1].child[2].split()
    t.child[2].child[1].split()
    return t
def make_example_tree3(with_data=False):
    if with_data:
        t = Node(dim=2,data_class=SimpleTest2D,prolongate_func=prolongate_datafunc,
                restrict_func=restrict_datafunc)
    else:
        t = Node(dim=2)
    t.split()
    for c in t.child:
        c.split()
        for x in c.child:
            x.split()
    return t
def make_example_tree_uniform(depth=3,with_data=False):
    if with_data:
        t = Node(dim=2,data_class=SimpleTest2D,prolongate_func=prolongate_datafunc,
                restrict_func=restrict_datafunc)
    else:
        t = Node(dim=2)
    for i in range(depth):
        for c in t.list_leaves():
            c.split()
    return t
def make_example_tree1(depth=2,with_data=False):
    if with_data:
        t = Node(dim=1,xmin=(-1,),xmax=(1,),data_class=SimpleTest1D,prolongate_func=prolongate_datafunc,
                restrict_func=restrict_datafunc)
    else:
        t = Node(dim=1)
    for i in range(depth):
        for c in t.list_leaves():
            c.split()


    return t

def func1(xc):
    """Function which sets the data value"""
    s = .1
    res = np.exp(-xc**2/(2*s**2))
    return res

def func(xc,yc):
    """Function which sets the data value"""
    cx = .65 + .5* 2.**(-8)
    cy = .65 +  .5* 2.**(-8)
    s = .1
    res = np.exp(-((xc-cx)**2+(yc-cy)**2)/(2*s**2))
    cx = .3 + .5* 2.**(-8)
    cy = .3 + .5* 2.**(-8)
    s = .1
    res += np.exp(-((xc-cx)**2+(yc-cy)**2)/(2*s**2))
    return res
class TestAMR():

    def test_clear_refine(self):
        t = Node(dim=2)
        t.split()
        for n in t.list_leaves():
            n.rflag = True
        clear_refine(t)

        assert all([n.rflag == False for n in t.list_leaves()])

    def test_compression(self):
        import io
        import sys
        t = make_example_tree()
        capturedOutput = io.StringIO()                  # Create StringIO object
        sys.stdout = capturedOutput                     #  and redirect stdout.
        compression(t)                                     # Call function.
        sys.stdout = sys.__stdout__                     # Reset redirect.
        ans = '31 points out of 8^2 = 64 for full grid\nYou have saved a factor of 2.06\nWith a compression factor of 51.56%\n'
        assert ans == capturedOutput.getvalue()

    def test_neighbor_check(self):
        t = make_example_tree()
        t.child[3].child[0].child[0].rflag = True
        t.child[1].child[2].rflag= False
        t.child[2].child[1].rflag= False
        neighbor_check(t.child[3].child[0].child[0])
        assert t.child[1].child[2].rflag
        assert t.child[2].child[1].rflag

    def test_start_refine(self):
        t = make_example_tree3(with_data=True)

        leaves = t.list_leaves()

        inds = [0,10,20,-1]
        for i in inds:
            leaves[i].rflag =True

        start_refine(t)


        assert all([l.rflag == False for l in leaves])
        for i in inds:
            assert leaves[i].child is not None
            assert leaves[i].leaf == False
    def test_start_derefine(self):
        t = make_example_tree3(with_data=True)

        leaves = t.list_leaves()

        inds = [0,10,20,-1]
        for i in inds:
            leaves[i].rflag =True

        start_refine(t)

        for i in inds:
            for c in leaves[i].child:
                c.rflag =True
        start_derefine(t)

        assert all([l.rflag == False for l in leaves])
        for i in inds:
            assert all([c is None for c in leaves[i].child])
            assert leaves[i].leaf == True

    def test_refine(self):
        t = make_example_tree3(with_data=True)

        leaves = t.list_leaves()

        rflags = [refinement_check(leaf,tol=.3)[0] for leaf in leaves]

        total_ans = len(list(filter(lambda x: x,rflags)))

        total = refine(t)

        assert total == total_ans

        for r,leaf in zip(rflags,leaves):
            if r:
                assert leaf.leaf == False
                assert leaf.child is not None

    def test_refinement_check(self):
        # Check that all neighbors get flagged
        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')

        final_list = get_refinement_neighbors(n,extent=2)

        res,value = refinement_check(n,tol=0,extent=2) # Guarentees a refinement

        assert n.rflag

        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')

        final_list = get_refinement_neighbors(n,extent=1)

        res,value = refinement_check(n,tol=0,extent=1) # Guarentees a refinement

        assert n.rflag


    def test_get_refinement_neighbors(self):
        # 1D
        t = make_example_tree1(depth=4,with_data=True)

        n = t.find('0x00x10x00x00x0')

        neighbors = ['0x00x00x10x10x0','0x00x00x10x10x1','0x00x10x00x00x0',
                    '0x00x10x00x00x1','0x00x10x00x10x0']

        assert neighbors == [x.name for x in get_refinement_neighbors(n)]


        # 2D
        # Extent = 1
        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')
        neighs = get_refinement_neighbors(n,extent=1)

        neighbors = ['0x00x00x30x3','0x00x10x20x2','0x00x10x20x3',
                    '0x00x20x10x1','0x00x30x00x0','0x00x30x00x1',
                    '0x00x20x10x3','0x00x30x00x2','0x00x30x00x3']
        print(neighbors)
        print([x.name for x in neighs])

        print([x.global_index[1:] for x in neighs])
        print([x.coords for x in neighs])
        assert neighbors == [x.name for x in neighs]

        # Extent = 2
        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')
        neighs = get_refinement_neighbors(n,extent=2)

        neighbors = ['0x00x00x30x0','0x00x00x30x1','0x00x10x20x0','0x00x10x20x1','0x00x10x30x0',
                     '0x00x00x30x2','0x00x00x30x3','0x00x10x20x2','0x00x10x20x3','0x00x10x30x2',
                    '0x00x20x10x0','0x00x20x10x1','0x00x30x00x0','0x00x30x00x1','0x00x30x10x0',
                    '0x00x20x10x2','0x00x20x10x3','0x00x30x00x2','0x00x30x00x3','0x00x30x10x2',
                    '0x00x20x30x0','0x00x20x30x1','0x00x30x20x0','0x00x30x20x1','0x00x30x30x0']
       # print(neighbors)
        for i,x in zip(neighbors,neighs):
            print(x.global_index[1:],i,x.name,i==x.name)


        #print([x.global_index[1:] for x in neighs])
       # print([x.coords for x in neighs])
        assert neighbors == [x.name for x in neighs]


    def test_refinement_flash(self):

        # 1D
        # Extent = 1
        uL=func1(-.125)
        uC = func1(0.)
        uR = func1(.125)
        num = (uL -2*uC+uR)**2
        den0 = abs(uR-uC)+abs(uL-uC)
        den1 = .01*(abs(uL)+2*abs(uC)+abs(uR))

        t = make_example_tree1(depth=4,with_data=True)
        n = t.find('0x00x10x00x00x0')
        neighs = get_refinement_neighbors(n,extent=1)
        minnum = 1e-4
        res,value = refinement_flash(n,neighs,tol=.5,eps=0,min_value=minnum)
        ans = np.sqrt(num/max(minnum,den0**2))
        # 1D eps=0 extent 1
        assert ans == value
        assert res == (ans>.5)

        res,value = refinement_flash(n,neighs,tol=.5,eps=0.01,min_value=minnum)
        ans = np.sqrt(num/max((den0+den1)**2,minnum))
        # 1D eps=.01 extent 1
        assert ans == value
        assert res == (ans>.5)


        # Extent = 2
        uL=func1(-.25)
        uC = func1(0.)
        uR = func1(.25)
        num = (uL -2*uC+uR)**2
        den0 = abs(uR-uC)+abs(uL-uC)
        den1 = .01*(abs(uL)+2*abs(uC)+abs(uR))

        t = make_example_tree1(depth=4,with_data=True)
        n = t.find('0x00x10x00x00x0')
        neighs = get_refinement_neighbors(n,extent=2)
        minnum = 1e-4
        res,value = refinement_flash(n,neighs,tol=.5,eps=0,min_value=minnum)
        ans = np.sqrt(num/max(minnum,den0**2))
        # 1D eps=0 extent 2
        assert ans == value
        assert res == (ans>.5)

        res,value = refinement_flash(n,neighs,tol=.5,eps=0.01,min_value=minnum)
        ans = np.sqrt(num/max((den0+den1)**2,minnum))
        # 1D eps=.01 extent 2
        assert ans == value
        assert res == (ans>.5)

        # 2D
        # Extent = 1
        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')
        neighs = get_refinement_neighbors(n,extent=1)
        u = np.zeros((3,3))



        u[0,0] = func(.375,.375)
        u[0,1] = func(.375,.5)
        u[0,2] = func(.375,.625)

        u[1,0] = func(.5,.375)
        u[1,1] = func(.5,.5)
        u[1,2] = func(.5,.625)

        u[2,0] = func(.625,.375)
        u[2,1] = func(.625,.5)
        u[2,2] = func(.625,.625)



        num = 0
        den = 0
        eps = 0
        minnum = 1e-4

        uC = u[1,1]
        uL = u[0,1]
        uR = u[2,1]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2
        uC = u[1,1]
        uL = u[1,0]
        uR = u[1,2]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2

        uC = u[1,1]
        uLL = u[0,0]
        uRR = u[2,2]
        uLR = u[2,0]
        uRL = u[0,2]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        uC = u[1,1]
        uLL = u[0,0]
        uRR = u[2,2]
        uLR = u[0,2]
        uRL = u[2,0]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        ans = np.sqrt(num/max(den,minnum))

        res,value = refinement_flash(n,neighs,tol=.5,eps=0,min_value=minnum)
        # 2D eps=0 extent 1
        assert abs(ans-value) < 1e-10
        assert res == (ans>.5)

        num = 0
        den = 0
        eps = 0.01
        minnum = 1e-4

        uC = u[1,1]
        uL = u[0,1]
        uR = u[2,1]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2
        uC = u[1,1]
        uL = u[1,0]
        uR = u[1,2]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2

        uC = u[1,1]
        uLL = u[0,0]
        uRR = u[2,2]
        uLR = u[2,0]
        uRL = u[0,2]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        uC = u[1,1]
        uLL = u[0,0]
        uRR = u[2,2]
        uLR = u[0,2]
        uRL = u[2,0]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2
        ans = np.sqrt(num/max(den,minnum))

        res,value = refinement_flash(n,neighs,tol=.5,eps=0.01,min_value=minnum)
        # 2D eps=.01 extent 1
        assert abs(ans-value) < 1e-10
        assert res == (ans>.5)


        # Extent = 2
        t = make_example_tree3(with_data=True)
        n = t.find('0x00x30x00x0')
        neighs = get_refinement_neighbors(n,extent=2)
        u = np.zeros((5,5))


        u[0,0] = func(.25,.25)
        u[0,1] = func(.25,.375)
        u[0,2] = func(.25,.5)
        u[0,3] = func(.25,.625)
        u[0,4] = func(.25,.75)

        u[1,0] = func(.375,.25)
        u[1,1] = func(.375,.375)
        u[1,2] = func(.375,.5)
        u[1,3] = func(.375,.625)
        u[1,4] = func(.375,.75)

        u[2,0] = func(.5,.25)
        u[2,1] = func(.5,.375)
        u[2,2] = func(.5,.5)
        u[2,3] = func(.5,.625)
        u[2,4] = func(.5,.75)

        u[3,0] = func(.625,.25)
        u[3,1] = func(.625,.375)
        u[3,2] = func(.625,.5)
        u[3,3] = func(.625,.625)
        u[3,4] = func(.625,.75)

        u[4,0] = func(.75,.25)
        u[4,1] = func(.75,.375)
        u[4,2] = func(.75,.5)
        u[4,3] = func(.75,.625)
        u[4,4] = func(.75,.75)


        num = 0
        den = 0
        eps = 0
        minnum = 1e-4

        uC = u[2,2]
        uL = u[0,2]
        uR = u[4,2]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2
        uC = u[2,2]
        uL = u[2,0]
        uR = u[2,4]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2

        uC = u[2,2]
        uLL = u[1,1]
        uRR = u[3,3]
        uLR = u[1,3]
        uRL = u[3,1]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        uC = u[2,2]
        uLL = u[1,1]
        uRR = u[3,3]
        uLR = u[3,1]
        uRL = u[1,3]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        ans = np.sqrt(num/max(den,minnum))

        res,value = refinement_flash(n,neighs,tol=.5,eps=0,min_value=minnum)
        # 2D eps=0 extent 2
        assert abs(ans-value) < 1e-10
        assert res == (ans>.5)

        num = 0
        den = 0
        eps = 0.01
        minnum = 1e-4

        uC = u[2,2]
        uL = u[0,2]
        uR = u[4,2]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2
        uC = u[2,2]
        uL = u[2,0]
        uR = u[2,4]
        num += (uL -2*uC+uR)**2
        den += (abs(uR-uC)+abs(uL-uC) + eps*(abs(uL)+2*abs(uC)+abs(uR)))**2

        uC = u[2,2]
        uLL = u[1,1]
        uRR = u[3,3]
        uLR = u[1,3]
        uRL = u[3,1]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2

        uC = u[2,2]
        uLL = u[1,1]
        uRR = u[3,3]
        uLR = u[3,1]
        uRL = u[1,3]
        num += (uLL + uRR - uLR - uRL)**2
        den += (abs(uLL-uRL) + abs(uRR-uLR) + eps*(abs(uLL)+abs(uRR)+abs(uRL)+abs(uLR)))**2
        ans = np.sqrt(num/max(den,minnum))

        res,value = refinement_flash(n,neighs,tol=.5,eps=0.01,min_value=minnum)
        # 2D eps=.01 extent 2
        assert abs(ans-value) < 1e-10
        assert res == (ans>.5)



