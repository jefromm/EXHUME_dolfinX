'''
Implementation of heat conduction problem with unfitted mesh
and Nitsche's method to weakly apply the BCs on a multi-material bar
'''


from mpi4py import MPI
from EXHUME_X.common import *
from dolfinx import mesh, fem, io
import os
import ufl
from petsc4py import PETSc
import numpy as np 
from timeit import default_timer

def unrotateAxisUFL(x,phiy,phiz):
    '''
    2D: performs rotation about origin
    3D: performs 2 rotations, about the z and y axis (in that order)
    switches axis from rotated orientation (in mesh) to align with bar

    Uses UFL operators to allow for automatic differentiation
    '''
    if dim == 2:
        phi = phiy
        x0 = x[0]*ufl.cos(-phi) - x[1]*ufl.sin(-phi)
        x1 = x[0]*ufl.sin(-phi) + x[1]*ufl.cos(-phi)
        return[x0,x1]
    
    x0 = x[0]*ufl.cos(-phiz) - x[1]*ufl.sin(-phiz)
    x1 = x[0]*ufl.sin(-phiz) + x[1]*ufl.cos(-phiz)
    x2 = x[2]
    x00 = x2*ufl.sin(-phiy) + x0*ufl.cos(-phiy)
    x11 = x1
    x22 = x2*ufl.cos(-phiy) - x0*ufl.sin(-phiy)
    return [x00,x11,x22]

def unrotateAxis(x,phiy,phiz):
    '''
    2D: performs rotation about origin
    3D: performs 2 rotations, about the z and y axis (in that order)
    switches axis from rotated orientation (in mesh) to align with bar
    
    Uses numpy operators to allow for interpolation
    '''
    if dim == 2:
        phi = phiy
        x0 = x[0]*np.cos(-phi) - x[1]*np.sin(-phi)
        x1 = x[0]*np.sin(-phi) + x[1]*np.cos(-phi)
        return[x0,x1]
    x0 = x[0]*np.cos(-phiz) - x[1]*np.sin(-phiz)
    x1 = x[0]*np.sin(-phiz) + x[1]*np.cos(-phiz)
    x2 = x[2]
    x00 = x2*np.sin(-phiy) + x0*np.cos(-phiy)
    x11 = x1
    x22 = x2*np.cos(-phiy) - x0*np.sin(-phiy)
    return [x00,x11,x22]

def rotateAxis(x,phiy,phiz):
    '''
    2D: performs rotation about origin
    3D: performs 2 rotations, about the y and z axis (in that order)
    switches axis from physical orientation (in bar) to align with mesh
    
    Uses numpy operators to allow for interpolation
    '''

    if dim == 2:
        phi = phiy
        x0 = x[0]*np.cos(phi) - x[1]*np.sin(phi)
        x1 = x[0]*np.sin(phi) + x[1]*np.cos(phi)
        return[x0,x1]
    x0 = x[2]*np.sin(phiy) + x[0]*np.cos(phiy)
    x1 = x[1]
    x2 = x[2]*np.cos(phiy) - x[0]*np.sin(phiy)
    x00 = x0*np.cos(phiz) - x1*np.sin(phiz)
    x11 = x0*np.sin(phiz) + x1*np.cos(phiz)
    x22 = x2
    return [x00,x11,x22]

    
def interface_T(T,q,domain,dS,jump,C_T=10):
    n = ufl.avg(w_2*ufl.FacetNormal(domain))
    const = ufl.avg(jump*T) * ufl.dot(custom_avg((kappa*ufl.grad(q)),kappa,domain),n)*dS
    adjconst = ufl.avg(jump*q) * ufl.dot(custom_avg((kappa*ufl.grad(T)),kappa,domain),n)*dS
    gamma = gamma_int(C_T, kappa, domain)
    pen = gamma*ufl.avg(jump*T)*ufl.avg(jump*q)*dS
    return pen +const - adjconst


def dirichlet_T(T,q,Td,domain,dS,weight,h=None, C_T=10):
    size = ufl.JacobianDeterminant(domain)
    if h is None:
        h = size**(1/dim)
    n = ufl.FacetNormal(domain)
    const = ufl.avg(weight*((T-Td)*kappa*ufl.inner(ufl.grad(q), n)))*dS
    adjconst = ufl.avg(weight*(q*kappa*ufl.inner(ufl.grad(T), n)))*dS
    gamma = C_T *kappa / h 
    pen = ufl.avg(weight*(gamma*q*(T-Td)))*dS
    return pen + const - adjconst


# functions to identify bar faces and material interfaces
# could also use mesh function tags but this is easier for the 4 material problem 
def Left(x):
    x_rotate = unrotateAxis(x,phiy,phiz)
    tol = 1e-8
    if dim == 2: 
        x_const = np.isclose(x_rotate[0],0.00,atol=tol)
        y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
        return np.logical_and(x_const, y_const)
    x_const = np.isclose(x_rotate[0],0.00,atol=tol)
    y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
    z_const = np.logical_and((x_rotate[2] <=  H+tol),(x_rotate[2] >= 0-tol))
    return np.logical_and(x_const, np.logical_and(y_const,z_const))

def Right(x):
    x_rotate = unrotateAxis(x,phiy,phiz)
    tol = 1e-8
    if dim ==2:
        x_const = np.isclose(x_rotate[0],L,atol=tol)
        y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
        return np.logical_and(x_const, y_const)
    x_const = np.isclose(x_rotate[0],L,atol=tol)
    y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
    z_const = np.logical_and((x_rotate[2] <=  H+tol),(x_rotate[2] >= 0-tol))
    return np.logical_and(x_const, np.logical_and(y_const,z_const))
def Interface12(x):
    x_rotate = unrotateAxis(x,phiy,phiz)
    tol = 1e-8
    if dim ==2:
        x_const = np.isclose(x_rotate[0],L/4,atol=tol)
        y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
        return np.logical_and(x_const, y_const)
    x_const = np.isclose(x_rotate[0],L/4,atol=tol)
    y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
    z_const = np.logical_and((x_rotate[2] <=  H+tol),(x_rotate[2] >= 0-tol))
    return np.logical_and(x_const, np.logical_and(y_const,z_const))
def Interface23(x):
    x_rotate = unrotateAxis(x,phiy,phiz)
    tol = 1e-8
    if dim == 2:
        x_const = np.isclose(x_rotate[0],3*L/4,atol=tol)
        y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
        return np.logical_and(x_const,y_const)
    x_const = np.isclose(x_rotate[0],3*L/4,atol=tol)
    y_const = np.logical_and((x_rotate[1] <=  H+tol),(x_rotate[1] >= 0-tol))
    z_const = np.logical_and((x_rotate[2] <=  H+tol),(x_rotate[2] >= 0-tol))
    return np.logical_and(x_const, np.logical_and(y_const,z_const))



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--of',dest='of',default="../../results.csv",
                    help='Output file destination')
parser.add_argument('--dim',dest='d',default=2,
                    help='dimension')
args = parser.parse_args()

k = int(args.k)
dim = int(args.d)
write_file=args.wf
if write_file == 'True':
    write_file = True
else:
    write_file = False
output_file = args.of

# check if there is a quad/hex mesh - may only have tris/tets for very coarse meshes 
quads = os.path.exists("meshes/quad.xdmf")
if quads:
    mesh_types = ["tri","quad"]
else:
    mesh_types = ["tri"]


ref = os.getcwd()[-1]
n = 2*3*(2**int(ref))
# note: over estimates by (k*n + 1)**(dim-1)
bg_dofs_guess = np.ceil(k * (2*n**dim + n*1.1))

if k == 2: 
    fluxFileWriter = outputVTX
else:
    fluxFileWriter = outputXDMF


L = 5.0
H = 1.0
phiy = -20*np.pi/180
phiz = 20*np.pi/180
if dim == 2:
    phiy = 20*np.pi/180
    phiz = None
exOpName = 'Elemental_Extraction_Operators_B0.hdf5'
filenames = [exOpName]
folder = "thermal_results/"
kappa_1= 1.0
kappa_2 = 0.1
kappa_3 = kappa_1 

#phase markers, from file
outside_ID = 0
bar_m1_ID = 1
bar_m2_ID = 2
bar_m3_ID = 3
# facet markers, user specified
left_ID = 1
right_ID = 2
interface_12_ID = 3
interface_23_ID = 4
facet_markers = [left_ID, right_ID, interface_12_ID,interface_23_ID]
facet_functions = [Left, Right, Interface12,Interface23]

num_facet_phases =len(facet_markers)


As =[]
bs = []
Ms = []
domains = []
T_exs =[]
T_ex_disps = []
Ts = []
t_exs = []
dx_bars = []

for subMeshType in mesh_types:
    
    meshFile = "meshes/" + subMeshType + ".xdmf"

    #Read in mesh and cell_map data 
    with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
        idt = xdmf.read_meshtags(domain, name="Grid")
        xdmf.close()

    materialMeshFile = "meshes/" + subMeshType + "_materials.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, materialMeshFile, "r") as xdmf:
        ct = xdmf.read_meshtags(domain, name="Grid")
        xdmf.close()
    cell_mat = ct.values


    dim_mesh = domain.topology.dim
    if dim != dim_mesh:
        dim = dim_mesh
        bg_dofs_guess = 2* (k*n + 1)**dim 

    outside_subdomain = ct.find(outside_ID)
    bar_m1_subdomain = ct.find(bar_m1_ID)
    bar_m2_subdomain = ct.find(bar_m2_ID)
    bar_m3_subdomain = ct.find(bar_m3_ID)
    bar_subdomain = np.concatenate((bar_m1_subdomain,bar_m2_subdomain,bar_m3_subdomain))

    domain.topology.create_connectivity(dim-1, dim)
    num_facets = domain.topology.index_map(dim-1).size_global
    f_to_c_conn = domain.topology.connectivity(dim-1,dim)
    facets = np.asarray([],dtype=np.int32)
    facets_mark = np.asarray([],dtype=np.int32)
    for phase in range(num_facet_phases):
        facets_phase = mesh.locate_entities(domain, dim-1, facet_functions[phase])
        facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
        facets= np.hstack((facets,facets_phase))
        facets_mark = np.hstack((facets_mark,facets_phase_mark))
    sorted_facets = np.argsort(facets)
    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])


    quadFactor = 4
    # define integration measurements for the domain of interest and the interior surface of interest
    dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 4*k})
    ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 4*k})
    dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 4*k})

    dx_bar = dx(bar_m1_ID)+dx(bar_m2_ID)+dx(bar_m3_ID)

    #create weight function to control integration on the interior surface 
    V_DG = fem.FunctionSpace(domain, ("DG", 0))

    w_1 = fem.Function(V_DG)
    w_2 = fem.Function(V_DG)
    w_3 = fem.Function(V_DG)
    jump = fem.Function(V_DG)
    kappa = fem.Function(V_DG)
    weights = [w_1, w_2, w_3,jump,kappa]
    weight_vals = [[2,0,0],\
                [0,2,0],\
                [0,0,2],\
                [-2,2,-2],\
                [kappa_1,kappa_2,kappa_3]]
    i = 0 
    for w in weights:
        w.x.array[bar_m1_subdomain] = weight_vals[i][0]
        w.x.array[bar_m2_subdomain] = weight_vals[i][1]
        w.x.array[bar_m3_subdomain] = weight_vals[i][2]
        w.x.array[outside_subdomain] = 0
        w.x.scatter_forward()
        i += 1 
    w_bar = w_1 + w_2 + w_3 

    # create solution function space
    V = fem.FunctionSpace(domain, ("DG", k))
    '''
    V_flux =  fem.FunctionSpace(domain, ("DG", k-1))
    # option to save material information with same function space
    # useful when using paraview to visualize in 3D
    outputVTX(w_bar, V, folder,'mat_'+subMeshType)
    fluxFileWriter(w_bar, V_flux, folder,'mat_flux'+subMeshType)
    '''
    T = fem.Function(V)
    q = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    def T_source_ufl(x): 
        x_bar = unrotateAxisUFL(x,phiy,phiz)[0]
        return ufl.sin(4*ufl.pi*x_bar/L)/kappa
    T_ex=T_source_ufl(x)
    def T_source_fun(x): 
        x_bar = unrotateAxis(x,phiy,phiz)[0]
        return np.sin(4*np.pi*x_bar/L)/kappa
    T_plot_ex =fem.Function(V)
    f = -ufl.div(ufl.grad(T_ex))*kappa 

    # specify residual
    res_T = kappa* ufl.inner(ufl.grad(q),ufl.grad(T))*(dx_bar) - ufl.inner(q,f)*(dx_bar)
    resD_T_l = dirichlet_T(T,q,T_ex,domain,dS(left_ID),w_1)
    resD_T_r = dirichlet_T(T,q,T_ex,domain,dS(right_ID),w_3)
    resI_12 = interface_T(T,q,domain,dS(interface_12_ID),jump,C_T=10)
    resI_23 = interface_T(T,q,domain,dS(interface_23_ID),jump,C_T=10)
    res = res_T + resD_T_l + resD_T_r +resI_12 +  resI_23 

    # Get Jacobian 
    J = ufl.derivative(res,T)

    # convert from dolfinx product objects to PETSc vector and matrix
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)

    # update ghost cells (needed when running in parallel)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    J_petsc.assemble()
    sizes = res_petsc.getSizes()

    # read extraction operator
    t_start = default_timer()
    M = readExOpElementwise(V,filenames,idt,domain,sizes,subdomain=bar_subdomain,type=subMeshType,bg_size=bg_dofs_guess)
    t_stop = default_timer()
    t_ex = t_stop-t_start
    # assemble background linear system
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)

    # save objects needed from both submesh types
    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    T_exs +=[T_ex]
    T_ex_disps += [T_plot_ex]
    Ts += [T]
    dx_bars += [dx_bar] 
    t_exs += [t_ex]

# if there are quad meshes:
if quads: 
    A_tri,A_quad = As
    b_tri,b_quad = bs
    M_tri,M_quad = Ms
    T_tri,T_quad = Ts
    t_ex_tri,t_ex_quad = t_exs

    # add the two matrices
    A_tri.axpy(1.0,A_quad)
    b_tri.axpy(1.0,b_quad)

    x = A_tri.createVecLeft()

    # solve the background linear system
    t_start = default_timer()
    solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')
    t_stop = default_timer()
    t_solve = t_stop-t_start
    
    # transfer to foreground
    transferToForeground(T_tri, x, M_tri)
    transferToForeground(T_quad, x, M_quad)

    # save solution 
    T_solns = [T_tri,T_quad]
else:
    A_tri = As[0]
    b_tri= bs[0]
    M_tri = Ms[0]
    T_tri = Ts[0]
    t_ex_tri = t_exs[0]
    t_ex_quad = 0 

    x = A_tri.createVecLeft()

    # solve background linear system
    t_start = default_timer()
    solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')
    t_stop = default_timer()
    t_solve = t_stop-t_start

    # transfer to foreground 
    transferToForeground(T_tri, x, M_tri)
    T_solns = [T_tri]

i = 0 
L2_net = 0 
H1_net = 0 
sum_net = 0 
L2norm_net = 0 
H1norm_net = 0 

# compute errors 
for subMeshType in mesh_types:
    T_soln = T_solns[i]
    domain = domains[i]
    T_ex = T_exs[i]
    T_ex_disp = T_ex_disps[i]
    dx_bar = dx_bars[i]

    T_sum = fem.assemble_scalar(fem.form(ufl.inner(T_soln , T_soln) * dx_bar))
    T_sum_assembled  = domain.comm.allreduce(T_sum, op=MPI.SUM)
    
    L2 = fem.assemble_scalar(fem.form(ufl.inner(T_soln-T_ex, T_soln-T_ex) * dx_bar))
    L2_assembled  = domain.comm.allreduce(L2, op=MPI.SUM)

    H1 = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(T_soln-T_ex), ufl.grad(T_soln-T_ex)) * dx_bar))
    H1_assembled  = domain.comm.allreduce(H1, op=MPI.SUM)

    L2_norm = fem.assemble_scalar(fem.form(ufl.inner(T_ex, T_ex) * dx_bar))
    L2_assembled_norm  = domain.comm.allreduce(L2_norm, op=MPI.SUM)

    H1_norm = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(T_ex), ufl.grad(T_ex)) * dx_bar))
    H1_assembled_norm  = domain.comm.allreduce(H1_norm, op=MPI.SUM)
    visualizeData = False
    if visualizeData:
        with io.VTXWriter(domain.comm, folder + "T_" + subMeshType + ".bp", [T], engine="BP4") as vtx:
            vtx.write(0.0)
        V_flux =  fem.FunctionSpace(domain, ("DG", k-1))
        q_mag_val = ufl.sqrt(ufl.dot(ufl.grad(T_soln),ufl.grad(T_soln)))
        fluxFileWriter(q_mag_val,V_flux,folder,"q_mag"+subMeshType)
        q_flux = ufl.grad(T_soln)
        fluxFileWriter(q_flux[0],V_flux,folder,"q_0"+subMeshType)
        fluxFileWriter(q_flux[1],V_flux,folder,"q_1"+subMeshType)
    i += 1
    L2_net += L2_assembled
    H1_net += H1_assembled
    sum_net += T_sum_assembled
    L2norm_net += L2_assembled_norm 
    H1norm_net += H1_assembled_norm 

L2_error = np.sqrt(L2_net / L2norm_net)
H10_error = np.sqrt(H1_net / H1norm_net)
sum_T = np.sqrt(sum_net)

# Only print the error on one process
if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"Error L2: {L2_error}")
    print(f"Error H10: {H10_error}")
    print(f"Sum T : {sum_T}")
    print(f"Extraction Time, tris : {t_ex_tri}")
    print(f"Extraction Time, quads : {t_ex_quad}")
    print(f"Solver Time : {t_solve}")

    if write_file: 
        f = open(output_file,'a')
        f.write("\n")
        fs =  str(MPI.COMM_WORLD.size)+","+str(ref)+","+str(k)+","+str(L2_error)+","+str(H10_error)+","+str(t_ex_tri)+","+str(t_ex_quad)+","+str(t_solve)
        f.write(fs)
        f.close()
