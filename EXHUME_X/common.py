from petsc4py import PETSc 
from dolfinx import cpp, la, fem , io 
import ufl
import numpy as np
import h5py  
from mpi4py import MPI
#note: need at least dolfinx version 0.5.0
#print(dolfinx.__version__)

import dolfinx
from dolfinx import mesh, fem, io, cpp 
from dolfinx.fem import petsc
from dolfinx.cpp import io as c_io
import os
import ufl
from petsc4py import PETSc
import numpy as np 
import argparse
from mpi4py import MPI

worldcomm = MPI.COMM_WORLD

def custom_avg(val, param, domain):
    #take average of the quantity  over a facet 
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    w_mag = h('+')/param('+') + h('-')/param('-')
    ave = (val('+')*h('+')/param('+') + val('-')*h('-')/param('-'))/w_mag 
    return ave 

def gamma_int(c, param, domain):
    # calculate penalty parameter based on 
    # bordering element sizes and relative parameter sizes
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    w_mag = h('+')/param('+') + h('-')/param('-')
    ave = 2*(c*h('+') + c*h('-'))/w_mag 
    return ave 

def getInterfaceFacets(f_to_c_conn, num_facets, cell_mat):
    """
    Creates list of facets on the interface of two material subdomains .

    Parameters
    -----------
    f_to_c_conn: facet to cell connectivity array 
    num_facets: number of mesh facets
    cell_mat: meshtag function identifying materials, assumes mat 1 marker = 1, mat 2 marker = 0 

    Returns
    ------ 
    interface_facets: np.array list of facet ids 
    """
    interface_facets = []
    for facet in range(num_facets):
        marker = 0
        cells = f_to_c_conn.links(facet)
        for cell in cells:
            marker = marker + cell_mat[cell] +1
        if marker == 3: 
            interface_facets += [facet]
    interface_facets = np.asarray(interface_facets,dtype=np.int32)
    return interface_facets

def outputXDMF(f,V,folder,name):
    ''' 
    function to interpolate a ufl object onto a function space, and then 
    plot the function on the function space's domain to visualize as an 
    xdmf file 
    '''
    domain = V.mesh
    f_expr = fem.Expression(f, V.element.interpolation_points())
    f_fun= fem.Function(V)
    f_fun.interpolate(f_expr)
    xdmf = io.XDMFFile(domain.comm, folder +name + ".xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(f_fun)

def outputVTX(f,V,folder,name):
    ''' 
    function to interpolate a ufl object onto a function space, and then 
    plot the function on the function space's domain to visualize as an 
    xdmf file 
    '''

    domain = V.mesh
    f_expr = fem.Expression(f, V.element.interpolation_points())
   
    f_fun= fem.Function(V)
    f_fun.interpolate(f_expr)
    with io.VTXWriter(domain.comm, folder+name+'.bp', [f_fun], engine="BP4") as vtx:
        vtx.write(0.0)




def solveKSP(A,b,u,method='gmres', PC='jacobi',
            rtol=1E-8,atol=1E-9, max_it=1000000,
            monitor=True,gmr_res=3000,):
    """
    solve linear system A*u=b
    A: PETSC Matrix
    b: PETSC Vector
    u: PETSC Vector
    """
    if method == None:
        method='gmres'
    if PC == None:
        PC='jacobi'

    if method == 'mumps':

        ksp = PETSc.KSP().create() 
        ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

        opts = PETSc.Options("mat_mumps_")
        # icntl_24: controls the detection of “null pivot rows", 1 for on, 0 for off
        # without basis function removal, this needs to be 1 to avoid crashing 
        opts["icntl_24"] = 1
        # cntl_3: is used to determine null pivot rows
        opts["cntl_3"] = 1e-12           

        A.assemble()
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc=ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setUp()

        ksp.solve(b,u)
        return 


    ksp = PETSc.KSP().create() 
    ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

    if method == 'gmres': 
        ksp.setType(PETSc.KSP.Type.FGMRES)
    elif method == 'gcr':
        ksp.setType(PETSc.KSP.Type.GCR)
    elif method == 'cg':
        ksp.setType(PETSc.KSP.Type.CG)

    if PC == 'jacobi':
        A.assemble()
        ksp.setOperators(A)
        pc = ksp.getPC()
        pc.setType("jacobi")
        ksp.setUp()
        ksp.setGMRESRestart(300)

    elif PC == 'ASM':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMOverlap(1)
        ksp.setUp()
        localKSP = pc.getASMSubKSP()[0]
        localKSP.setType(PETSc.KSP.Type.FGMRES)
        localKSP.getPC().setType("lu")
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ICC':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("icc")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ILU':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("euclid")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC == 'ILUT':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("pilut")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["ksp_view"] = None
    ksp.setFromOptions()
    ksp.solve(b,u)
    
    history = ksp.getConvergenceHistory()
    if monitor:
        print('Converged in', ksp.getIterationNumber(), 'iterations.')
        print('Convergence history:', history)


def  assembleLinearSystemBackground(a_f, L_f, M):
    """
    Assemble the linear system on the background mesh, with
    variational forms defined on the foreground mesh.
    
    Parameters
    ----------
    a_f: LHS PETSc matrix
    L_f: RHS PETSc matrix
    M: extraction petsc matrix 
    
    Returns
    -------  
    A_b: PETSc matrix on the background mesh
    b_b: PETSc vector on the background mesh
    """

    A_b = AT_R_A(M, a_f)
    b_b = AT_x(M, L_f)
    return A_b, b_b

def AT_R_A(A, R):
    """
    Compute "A^T*R*A". A,R are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat

    Returns
    ------ 
    ATRA : petsc4py.PETSc.Mat
    """
    AT = A.transpose()
    ATR = AT.matMult(R)
    ATT = A.transpose()
    ATRA = ATR.matMult(ATT)
    return ATRA


def AT_x(A, x):
    """
    Compute b = A^T*x.
    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    x : petsc4py.PETSc.Vec
    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    
    b_PETSc = A.createVecRight()
    A.multTranspose(x, b_PETSc)
    return b_PETSc

def transferToForeground(u_f, u_b, M):
    """
    Transfer the solution vector from the background to the forground
    mesh.
    
    Parameters
    ----------
    u_f: Dolfin function on the foreground mesh
    u_b: PETSc vector of soln on the background mesh 
    M: extraction matrix from background to foreground.
    """
    #u_petsc = cpp.la.petsc.create_vector_wrap(u_f.x)
    u_petsc = la.create_petsc_vector_wrap(u_f.x)
    M.mult(u_b, u_petsc)
    u_f.x.scatter_forward()

def readExOpElementwise(W,filenames,idt,domain,sizes,
                        subdomain=None,factor= 0.5,
                        type ='tri',bg_size=None):
    """
    Creates extraction matrix from file data .
    
    Parameters
    ----------
    W: FG function space
    filenames: T-matrix file names, one per scalar field in the FG space
    idt: mesh function giving each cell's T-matrix id 
    domain: mesh 
    sizes: FG local and global matrix sizes, to preallocate M
    subdomain: mesh subdomain region, defaults to entire mesh
    factor: scale factor relating number of assumed BG dofs to FG dofs, default is 0.5 assuming DG space
    type: mesh type, defaults to 'tri'
    bg_size: guess of number of BG degrees of freedom 

    Returns
    ----------
    M: extraction matrix from background to foreground.
    """
    dim = domain.topology.dim
    weight_key_base = 'Weights_'
    id_key_base = 'IDs_'

    cell_map= idt.values # vector, index is dolfinx cell ID, value is moris cell ID
    local_cell_indices = idt.indices

    cell_index_map = domain.topology.index_map(domain.topology.dim)
    num_cells_local = cell_index_map.size_local

    local_size = sizes[0]
    global_size = sizes[1]
    num_fg_dofs = global_size
    if bg_size is not None:
        num_bg_dofs = bg_size
    else:
        num_bg_dofs = np.ceil(factor* num_fg_dofs) 
    
    M = PETSc.Mat().create(comm=worldcomm) 
    M.setSizes(((local_size, global_size), num_bg_dofs))
    M.setUp()
    no_fields = len(filenames)
    num_bg_dofs_per_field = np.ceil(num_bg_dofs/no_fields)

    for field in range(no_fields):
        max_bg_ID = 0
        filename = filenames[field]
        if no_fields == 1:
            dofmap = W.dofmap
        else:
            dofmap = W.sub(field).dofmap 
        dofmapIndexMap = dofmap.index_map
        with h5py.File(filename, "r") as f:
            cell_count = 0 
            if subdomain is not None:
                indices = subdomain
            else:
                indices = range(num_cells_local)
            for local_cell_index in indices: 
                if local_cell_index < num_cells_local: # this means the cell is not a ghost cell,
                    cell_count+= 1
                    #local fg_IDs
                    fg_IDs = dofmap.cell_dofs(local_cell_index)
                    ID = cell_map[local_cell_index]
                    # global bg_IDs
                    bg_IDs = np.asarray(f[id_key_base+str(ID)])
                    cell_max_bg_ID = max(bg_IDs[0])
                    if cell_max_bg_ID> max_bg_ID:
                        max_bg_ID = cell_max_bg_ID
                    ex_weights = np.asarray(f[weight_key_base+str(ID)])
                    # loop over fgs in cell 
                    fg_count = 0 
                    #note: (fenics,exo) tri6 node map: 
                    #      (0,0), (1,1), (2,2), (3,4), (4,5) (5,3)
                    if type == 'tri':
                        if dim == 2:  
                            # high order mapping from fenics scheme to exo scheme
                            fenics2morisLocalNodes = [0,1,2,4,5,3]
                        else:
                            #high order mapping from fenics scheme to exo scheme
                            fenics2morisLocalNodes = [0,1,2,3,9,8,5,7,6,4]
                    elif type =='quad':
                        if dim == 2: 
                            fenics2morisLocalNodes = [0,1,3,2,4,7,5,6,8] # correct when f = x[0]
                        else:
                            print("only 2D mixed meshes supported")
                            exit()
                    else:
                        print("only tris or quads supported")
                        exit()
                    for fg_ID in fg_IDs:
                        bg_count = 0 
                        for bg_ID in bg_IDs[0]:
                            morisFGID = fenics2morisLocalNodes[fg_count]
                            weight = ex_weights[morisFGID][bg_count]
                            #note: addv=2 corresponds to ADD_VALUES
                            #note: addv=1 corresponds to INSERT_VALUES
                            glo_fg_ID = dofmapIndexMap.local_to_global([fg_ID])
                            M.setValue(glo_fg_ID, (bg_ID+field*num_bg_dofs_per_field), weight,addv=1)
                            bg_count += 1
                        fg_count += 1 
    M.assemble()
    return M

def L2Project(u_p, u_f, expression_f, M,dx_= None,bfr_tol=None):
    """
    Project the UFL expression of the initial condition
    onto the function spaces of the foreground mesh and 
    the background mesh, to make u_f = M*u_p.
    
    Parameters
    -----------
    u_p: PETSc Vector representing the dofs on the background mesh
    u_f: Dolfin Function on the foreground mesh
    expression_f: initial condition expression on the foreground
    M: extraction matrix from the background to the foreground
    """
    if dx_ == None:
        dx_ = ufl.dx
    V_f = u_f.function_space()
    u_f_0 = ufl.TrialFunction(V_f)
    w_f = ufl.TestFunction(V_f)
    a_f = ufl.inner(u_f_0, w_f)*dx_
    L_f = ufl.inner(expression_f, w_f)*dx_
    A_b, b_b = assembleLinearSystemBackground(a_f,L_f,M)
    solveKSP(A_b,b_b,u_p,monitor=False)
    transferToForeground(u_f, u_p, M)