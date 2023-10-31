'''
Implementation of 2D Poisson's problem with unfitted mesh
and Nitsche's method to weakly apply the BCs

Based on the Poisson problem demo from the dolfinx tutorial:
https://jorgensd.github.io/dolfinx-tutorial/chapter1/fundamentals_code.html
'''

from EXHUME_X.common import *

def interiorResidual(u,v,f,mesh,dx_,ds_,domain_weights):
    n = ufl.FacetNormal(mesh)
    if ds_ is not None:
        return ufl.inner(ufl.grad(u), ufl.grad(v))*dx_ \
            - ufl.avg(domain_weights*ufl.inner(ufl.dot(ufl.grad(u), n), v))*ds_ \
            - ufl.inner(f, v)*dx_
    else:
        return ufl.inner(ufl.grad(u), ufl.grad(v))*dx_ \
            - ufl.inner(f, v)*dx_


def boundaryResidual(u,v,u_exact,mesh, ds_, domain_weights,
                        sym=True,
                        beta_value=10,
                        overPenalize=False,
                        h=None):

    '''
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/poisson_circle_dirichlet.py
    '''
    n = ufl.FacetNormal(mesh)
    if h is not None:
        h_E = h
    else:
        size = ufl.avg(domain_weights*ufl.JacobianDeterminant(mesh))
        dim = mesh.topology.dim
        if dim == 2:
            h_E = size**(0.5)
        elif dim == 3:
            h_E = size**(1/3)
    beta =fem.Constant(domain, PETSc.ScalarType(beta_value))
    if sym:
        sgn = 1.0
    else:
        sgn = -1.0
    retval = sgn*ufl.avg(domain_weights*ufl.inner(u_exact-u, ufl.dot(ufl.grad(v), n)))*ds_ 
    penalty = beta*h_E**(-1)*ufl.avg(domain_weights*ufl.inner(u-u_exact, v))*ds_
    if (overPenalize or sym):
        retval += penalty
    return retval

def u_exact_fun(x): 
    return  np.sin(x[1] + x[0]+ 0.1)*np.cos(x[1] - x[0]- 0.1)
def u_exact_ufl(x): 
    return ufl.sin(x[1] + x[0]+ 0.1)*ufl.cos(x[1] - x[0]- 0.1)

import argparse
parser = argparse.ArgumentParser()               
parser.add_argument('--k',dest='k',default=1,
                    help='FG polynomial degree.')
parser.add_argument('--kHat',dest='kHat',default=None,
                    help='Background spline polynomial degree, default to FG degree ku')
parser.add_argument('--spline',dest='spline',default=None,
                    help='Background spline T-matrices file')
parser.add_argument('--mode',dest='mode',default='strain',
                    help='strain or stress, refering to plane strain or plane stress (default is strain)')
parser.add_argument('--lr',dest='lr',default=0,
                    help='level of local refinement, for data reporting, default 0')
args = parser.parse_args()

k = int(args.k)
kHat=args.kHat
spline=args.spline
lr = int(args.lr)
mode = args.mode

folder = 'poisson_results/'

if kHat is None:
    kHat = k 
else:
    kHat = int(args.kHat)

if spline == None:
    if kHat == 1:
        exOpName = 'Elemental_Extraction_Operators_B0.hdf5'
    else:
        exOpName = 'Elemental_Extraction_Operators_B1.hdf5'
else: 
    exOpName = spline
filenames = [exOpName]

# guess number of bg dofs to pre allocate the matrix size 
ref = os.getcwd()[-1]
n = 8*(2**int(ref))

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
bg_dofs_guess = np.ceil(kHat * (2*n**2 + n*1.1))

# cell markers, from mesh file
inside_ID = 1
outside_ID = 0

# facet markers, user specified
interface_ID = 1


As =[]
bs = []
Ms = []
domains = []
u_exs =[]
u_ex_disps = []
us = []
dxs = []

if lr >= 1: 
    # we use a for loop to define our linear algebra objects for each submesh 
    # for visualization, we also need to save each submeshes function space and material parameters 
    mesh_types = ["tri","quad"]
else: 
    # no local refinement, assume single tri mesh 
    mesh_types = ["tri"]

for subMeshType in mesh_types:
    #Read in mesh and cell_map data 
    meshFile = "meshes/" + subMeshType + ".xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
        idt = xdmf.read_meshtags(domain, name="Grid")
        xdmf.close()
    materialMeshFile = "meshes/" + subMeshType + "_materials.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, materialMeshFile, "r") as xdmf:
        ct = xdmf.read_meshtags(domain, name="Grid")
        xdmf.close()
    cell_mat = ct.values

    inside_subdomain = ct.find(inside_ID)
    outside_subdomain  = ct.find(outside_ID)

    dim = domain.topology.dim
    domain.topology.create_connectivity(dim-1, dim)

    num_facets = domain.topology.index_map(dim-1).size_local
    f_to_c_conn = domain.topology.connectivity(dim-1,dim)

    facets = getInterfaceFacets(f_to_c_conn, num_facets, cell_mat)
    facets_mark = interface_ID*np.ones_like(facets)

    sorted_facets = np.argsort(facets)
    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])


    # define integration measurements for the domain of interest and the interior surface of interest
    dx_custom = ufl.Measure('dx',subdomain_data=ct,subdomain_id=inside_ID,metadata={'quadrature_degree': 2*k})
    ds_custom = ufl.Measure('dS',domain=domain, subdomain_data=ft,subdomain_id=interface_ID,metadata={'quadrature_degree': 2*k})

    V_DG = fem.FunctionSpace(domain, ("DG", 0))
    weight = fem.Function(V_DG)
    weight.x.array[inside_subdomain] = 2
    weight.x.array[outside_subdomain] = 0
    weight.x.scatter_forward()

    #define variational problem 
    V = fem.FunctionSpace(domain, ("DG", k))
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    #define source term 

    # as a function for visualization
    u_ex_disp = fem.Function(V)
    u_ex_disp.interpolate(u_exact_fun)

    # as a ufl object for the body force and BC
    x = ufl.SpatialCoordinate(domain)
    u_ex = u_exact_ufl(x) 
    f = -ufl.div(ufl.grad(u_ex))

    res_interior = interiorResidual(u, v, f, domain,dx_custom,ds_custom, weight)
    res_boundary = boundaryResidual(u, v, u_ex, domain,ds_custom,weight, sym=True,beta_value=10.0)
    res = res_interior + res_boundary

    J = ufl.derivative(res,u)

    res_petsc = fem.petsc.assemble_vector(fem.form(res))
    J_petsc = fem.petsc.assemble_matrix(fem.form(J))

    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    sizes = res_petsc.getSizes()

    M = readExOpElementwise(V,filenames,idt,domain,sizes,type =subMeshType,subdomain=inside_subdomain,bg_size=bg_dofs_guess)
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)

    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    u_exs +=[u_ex]
    u_ex_disps += [u_ex_disp]
    us += [u]
    dxs += [dx_custom] 

if lr>= 1:
    A_tri,A_quad = As
    b_tri,b_quad = bs
    M_tri,M_quad = Ms
    u_tri,u_quad = us
    # add the two matrices
    A_tri.axpy(1.0,A_quad)
    b_tri.axpy(1.0,b_quad)
    x = A_tri.createVecLeft()
    solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')

    transferToForeground(u_tri, x, M_tri)
    transferToForeground(u_quad, x, M_quad)
    u_solns = [u_tri,u_quad]
 
else: 
    A = As[0]
    b = bs[0]
    M = Ms[0]
    u = us[0]
    x = A.createVecLeft()
    solveKSP(A,b,x,monitor=False,method='mumps')
    transferToForeground(u, x, M)
    u_solns = [u]

L2s = []
H1s = []

i = 0 
for subMeshType in mesh_types:
    u = u_solns[i] 
    u_ex = u_exs[i]
    u_disp = u_ex_disps[i] 
    dx = dxs[i]
    domain = domains[i]

    L2_error = fem.form(ufl.inner(u - u_ex, u - u_ex) * dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    H10_error = fem.form(ufl.inner(ufl.grad(u - u_ex), ufl.grad(u - u_ex)) *dx)
    error_local = fem.assemble_scalar(H10_error)
    error_H10 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    with io.VTXWriter(domain.comm, folder +subMeshType +"_u.bp", [u], engine="BP4") as vtx:
        vtx.write(0.0)
    with io.VTXWriter(domain.comm, folder +subMeshType +"_u_ex.bp", [u_ex_disp], engine="BP4") as vtx:
        vtx.write(0.0)
    
    L2s += [error_L2]
    H1s += [error_H10]
    i+=1 


L2_net = sum(L2s)
H1_net = sum(H1s)

# Only print the error on one process
if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"L2 Error): {L2_net}")
    print(f"H10 Error: {H1_net}")