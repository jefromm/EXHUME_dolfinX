'''
Implementation of 2D linear elasticity problem
with Nitsche's method to weakly apply the BCs
'''

from mpi4py import MPI
from EXHUME_X.common import *
from dolfinx import mesh, fem, io, cpp 
import os
import ufl
from petsc4py import PETSc
import numpy as np 
from timeit import default_timer
import argparse


def epsU(u):
    '''
    Total Strain
    '''
    return ufl.sym(ufl.grad(u))
def sigma(eps):
    '''
    Stress 
    '''
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(u_dim)

def u_exact_ufl_in(x): 
    '''
    Exact solution inside circular inclusion
    defined using UFL operators for automatic differentiation
    '''
    r = (x[0]**2 + x[1]**2)**0.5
    theta = ufl.atan(x[1]/x[0])
    ur = C1*r
    ux = ur*ufl.cos(theta)
    uy = ur*ufl.sin(theta)
    return ufl.as_vector([ux,uy])

def u_exact_ufl_BC(x): 
    '''
    Exact solution outside circular inclusion
    defined using UFL operators for automatic differentiation
    '''
    r = ufl.operators.max_value(R/2,(x[0]**2 + x[1]**2)**0.5)
    theta = ufl.atan(x[1]/x[0])
    ur = C1*R*R/(r + 1e-8)
    ux = ur*ufl.cos(theta)
    uy = ur*ufl.sin(theta)
    return ufl.as_vector([ux,uy])

def ur_exact_ufl_in(x): 
    '''
    Exact solution for radial displacemebnt inside circular inclusion
    defined using UFL operators for automatic differentiation
    '''
    r = (x[0]**2 + x[1]**2)**0.5
    ur = C1*r
    return ur

def ur_exact_ufl_BC(x): 
    '''
    Exact solution for radial displacement outside circular inclusion
    defined using UFL operators for automatic differentiation
    '''
    r = ufl.operators.max_value(R/2,(x[0]**2 + x[1]**2)**0.5)
    ur = C1*R*R/(r)
    return ur

def dirichlet_u(u,v,ud,domain,ds,sgn,C_u=10,eps0= None):
    n = ufl.FacetNormal(domain)
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    const = sgn*ufl.inner(ufl.dot(sig_v, n), (u-ud))*ds
    adjconst = ufl.inner(ufl.dot(sig_u, n), v)*ds
    gamma = C_u *E /h 
    pen = gamma*ufl.inner(v,(u-ud))*ds
    return  pen -const - adjconst 

def symmetry_u(u,v,g,domain,ds,sgn,C_u=10,eps0 = None):
    beta = C_u*mu
    n = ufl.FacetNormal(domain)
    size = ufl.JacobianDeterminant(domain)
    h_E = size**(0.5)
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    nitsches_term =  -sgn*ufl.dot(ufl.dot(ufl.dot(sig_v,n),n),(ufl.dot(u,n)- g))*ds - ufl.dot(ufl.dot(ufl.dot(sig_u,n),n),ufl.dot(v,n))*ds
    penalty_term = beta*(h_E**(-1))*ufl.dot((ufl.dot(u,n)-g),ufl.dot(v,n))*ds
    return nitsches_term + penalty_term 

    
def interface_u(u,v,domain,dS,jump,C_u=10, eps0 = None):
    n = ufl.avg(w_inside*ufl.FacetNormal(domain))
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    const = ufl.inner(ufl.avg(jump*u),ufl.dot(custom_avg((sig_v),E,domain),n))*dS
    adjconst = ufl.inner(ufl.avg(jump*v),ufl.dot(custom_avg((sig_u),E,domain),n))*dS
    gamma = gamma_int(C_u, E, domain)
    pen = gamma*ufl.inner(ufl.avg(jump*u),ufl.avg(jump*v))*dS
    return const - adjconst + pen 

# functions defining exterior boundary locations
def Left(x):
    return np.isclose(x[0], 0)
def Right(x):
    return np.isclose(x[0], L)
def Top(x):
    return np.isclose(x[1], L)
def Bottom(x):
    return np.isclose(x[1], 0)

parser = argparse.ArgumentParser()
parser.add_argument('--k',dest='k',default=1,
                    help='Foreground polynomial degree.')
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')                   
parser.add_argument('--wf',dest='wf',default=False,
                    help='Write error to file (True/False)')
parser.add_argument('--of',dest='of',default="../results.csv",
                    help='Output file destination')
parser.add_argument('--path',dest='path',default=None,
                    help='Output file destination')
parser.add_argument('--mesh',dest='mesh',default='mesh.xdmf',
                    help='mesh file (overrides ref level)')
parser.add_argument('--mode',dest='mode',default='strain',
                    help='strain or stress, refering to plane strain or plane stress (default is strain)')
parser.add_argument('--kHat',dest='kHat',default=1,
                    help='Background spline polynomial degree, default to FG degree k')
parser.add_argument('--lr',dest='lr',default=0,
                    help='level of local refinement, for data reporting, default 0')
args = parser.parse_args()

write_file=args.wf
if write_file == 'True':
    write_file = True
else:
    write_file = False
output_file = args.of
k = int(args.k)
kHat=args.kHat
if kHat is None:
    kHat = k 
else:
    kHat = int(args.kHat)
m_path  = args.path
meshFile = args.mesh
sym = args.symmetric
if sym == 'False':
    sym = False
else:
    sym = True
mode = args.mode
if mode == 'strain':
    plane_stress = False
elif mode == 'stress':
    plane_stress = True
else: 
    print("only modes available are stress and strain")
    exit()
lr = int(args.lr)

# guess number of bg dofs to pre allocate the matrix size 
ref = os.getcwd()[-1]
n = 8*(2**int(ref))
no_fields = 2
bg_dofs_guess = no_fields*np.ceil(k * (3*((n+1)**2) + n*0.1))
meshFileName = args.mesh
exOpName = 'Elemental_Extraction_Operators_B0.hdf5'
g = 0.0
if sym:
    sgn = 1.0
else:
    sgn = -1.0
# Domain geometry information
L = 5
R = 1.0
mu_inside = 390.63
mu_outside = 338.35
lam_inside = 497.16
lam_outside = 656.79
eps0 = 0.1
C1 = (lam_inside + mu_inside)*eps0/(lam_inside + mu_inside + mu_outside) 

# cell markers, from mesh file
inside_ID = 0
outside_ID = 1

# facet markers, user specified
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4

u_dim = 2

facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]
num_facet_phases =len(facet_markers)



mesh_types = ["tri","quad"]
Vs = []
As =[]
bs = []
Ms = []
domains = []
u_exs =[]
ur_exs = []
us = []
t_exs = []
dxs = []

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

    inside_subdomain = ct.find(inside_ID)
    outside_subdomain  = ct.find(outside_ID)

    dim = domain.topology.dim
    domain.topology.create_connectivity(dim-1, dim)
    num_facets = domain.topology.index_map(dim-1).size_global
    f_to_c_conn = domain.topology.connectivity(dim-1,dim)
    interface_facets = []
    interface_facet_marks = []
    for facet in range(num_facets):
        marker = 0
        cells = f_to_c_conn.links(facet)
        for cell in cells:
            marker = marker + cell_mat[cell] +1
        if marker == 3: 
            interface_facets += [facet]
            interface_facet_marks += [interface_ID]
    interface_facets = np.asarray(interface_facets,dtype=np.int32)
    interface_facet_marks = np.asarray(interface_facet_marks,dtype=np.int32)

    # mark exterior boundaries using FEniCS function

    num_facet_phases =len(facet_markers)
    facets = np.asarray([],dtype=np.int32)
    facets_mark = np.asarray([],dtype=np.int32)
    for phase in range(num_facet_phases):
        facets_phase = mesh.locate_entities(domain, domain.topology.dim-1, facet_functions[phase])
        facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
        facets= np.hstack((facets,facets_phase))
        facets_mark = np.hstack((facets_mark,facets_phase_mark))

    # add interface facets to list
    facets= np.hstack((facets,interface_facets))
    facets_mark = np.hstack((facets_mark,interface_facet_marks))
    sorted_facets = np.argsort(facets)

    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])


    #create weight function to control integration on the interior surface 
    V_DG = fem.FunctionSpace(domain, ("DG", 0))
    w_inside = fem.Function(V_DG)
    w_inside.x.array[inside_subdomain] = 2
    w_inside.x.array[outside_subdomain] = 0
    w_inside.x.scatter_forward()
    w_outside = fem.Function(V_DG)
    w_outside.x.array[inside_subdomain] = 0
    w_outside.x.array[outside_subdomain] = 2
    w_outside.x.scatter_forward()

    # create DG function to calculate the jump 
    # Using the notation from (Schmidt 2023), the interior is material m and the exterior is n 
    # jump == [[.]] = (.)^m - (.)^n 
    jump = fem.Function(V_DG)
    jump.x.array[inside_subdomain] = 2
    jump.x.array[outside_subdomain] = -2
    jump.x.scatter_forward()

    # define integration measurements for the domain of interest and the interior surface of interest
    dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 2*k})
    ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})
    ds_exterior = ds(right_ID) + ds(left_ID) + ds(top_ID) + ds(bottom_ID)
    dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})

    # define solution function space
    el = ufl.FiniteElement("DG", domain.ufl_cell(),k)
    mel = ufl.MixedElement([el, el])
    V = fem.FunctionSpace(domain, mel)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    mu = fem.Function(V_DG)
    mu.x.array[inside_subdomain] = mu_inside
    mu.x.array[outside_subdomain] = mu_outside
    mu.x.scatter_forward()

    lam = fem.Function(V_DG)
    lam.x.array[inside_subdomain] = lam_inside
    lam.x.array[outside_subdomain] = lam_outside
    lam.x.scatter_forward()

    if plane_stress:
        lam = 2*mu*lam/(lam+2*mu)

    E = mu*(3*lam+2*mu)/(lam+mu) 

    u_ex = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    u_exact_in = u_exact_ufl_in(x) 
    u_exact_BC = u_exact_ufl_BC(x) 
    ur_exact_in = ur_exact_ufl_in(x) 
    ur_exact_BC = ur_exact_ufl_BC(x) 

    # create indicator function 
    inside = fem.Function(V_DG)
    inside.x.array[inside_subdomain] = 1.0
    inside.x.array[outside_subdomain] = 0.0
    inside.x.scatter_forward()
    eps0_tensor = eps0* ufl.as_tensor([[1,0],[0,1]])*inside
    u_exact = u_exact_in*inside - u_exact_BC*(inside - 1.0)
    ur_exact = ur_exact_in*inside - ur_exact_BC*(inside - 1.0)
    # define residuals 
    epsE = epsU(u) - eps0_tensor
    epsE_v = epsU(v)
    res_u = ufl.inner(epsU(v),sigma(epsE))*(dx(inside_ID) + dx(outside_ID))
    res_sym = symmetry_u(u,v,g,domain,(ds(bottom_ID) + ds(left_ID)),sgn,eps0=eps0_tensor) 
    resD_u_t = dirichlet_u(u,v,u_exact_BC,domain,ds(top_ID),sgn,eps0=eps0_tensor)
    resD_u_r = dirichlet_u(u,v,u_exact_BC,domain,ds(right_ID),sgn,eps0=eps0_tensor)
    resI_u = interface_u(u,v,domain,dS(interface_ID),jump,C_u=100,eps0=eps0_tensor)
    res = res_u + res_sym + resD_u_t + resD_u_r + resI_u 

    # define Jacobian 
    J = ufl.derivative(res,u)

    # convert from dolfinx products to PETSc vector / matrix
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)
    J_petsc.assemble()
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u_petsc = J_petsc.createVecLeft()
    
    sizes = res_petsc.getSizes()
    # read extraction operator
    t_start = default_timer()
    filenames = [exOpName, exOpName]
    M = readExOpElementwise(V,filenames,idt,domain,sizes,type =subMeshType,bg_size=bg_dofs_guess)
    t_stop = default_timer()
    t_ex = t_stop-t_start

    # assemble Background linear system 
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)
    
    # save quantities needed for both submesh types
    Vs += [V]
    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    u_exs +=[u_exact]
    ur_exs +=[ur_exact]
    us += [u]
    dxs += [dx] 
    t_exs += [t_ex]


A_tri,A_quad = As
b_tri,b_quad = bs
M_tri,M_quad = Ms
domain_tri,domain_quad = domains
u_ex_tri,u_ex_quad = u_exs 
u_tri,u_quad = us
dx_tri,dx_quad = dxs
t_ex_tri,t_ex_quad = t_exs

# add the two matrices/ vectors
A_tri.axpy(1.0,A_quad)
b_tri.axpy(1.0,b_quad)
x = A_tri.createVecLeft()

# solve background linear system 
t_start = default_timer()
solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')
t_stop = default_timer()
t_solve = t_stop-t_start

# transfer to foreground 
transferToForeground(u_tri, x, M_tri)
transferToForeground(u_quad, x, M_quad)


# compute errors
L2_error = fem.form(ufl.inner(u_tri - u_ex_tri , u_tri  - u_ex_tri ) * dx_tri)
error_local = fem.assemble_scalar(L2_error)
error_L2_tri  = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

H10_error = fem.form(ufl.inner(ufl.grad(u_tri  - u_ex_tri ), ufl.grad(u_tri  - u_ex_tri )) *dx_tri )
error_local = fem.assemble_scalar(H10_error)
error_H10_tri  = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

L2_error = fem.form(ufl.inner(u_quad - u_ex_quad , u_quad  - u_ex_quad ) * dx_quad)
error_local = fem.assemble_scalar(L2_error)
error_L2_quad  = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

H10_error = fem.form(ufl.inner(ufl.grad(u_quad  - u_ex_quad ), ufl.grad(u_quad  - u_ex_quad)) *dx_quad )
error_local = fem.assemble_scalar(H10_error)
error_H10_quad  = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

net_L2 = error_L2_tri + error_L2_quad
net_H10 = error_H10_tri + error_H10_quad


if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"Error_L2: {net_L2}")
    print(f"Error_H10: {net_H10}")
    print(f"Extraction Time (tris): {t_ex_tri}")
    print(f"Extraction Time (quads): {t_ex_quad}")
    print(f"Solver Time : {t_solve}")
    if write_file:
        print("writing file") 
        print(output_file)
        f = open(output_file,'a')
        f.write("\n")
        fs =  str(lr)+","+str(ref)+","+str(k)+","+str(net_L2)+","+str(net_H10)\
                +","+str(error_L2_tri)+","+str(error_H10_tri)\
                +","+str(error_L2_quad)+","+str(error_H10_quad)\
                +","+str(t_ex_tri)+","+str(t_ex_quad)+","+str(t_solve) 
        f.write(fs)
        f.close()

visualizeData = False
if visualizeData:
    u_solns = [u_tri,u_quad]
    if k == 2: 
        strainFileWriter = outputVTX
    else:
        strainFileWriter = outputXDMF

    VMs = []
    i = 0 
    for subMeshType in mesh_types:
        V = Vs[i] 
        u = u_solns[i] 
        u_exact = u_exs[i]
        ur_exact = ur_exs[i]
        domain = domains[i]

        folder = "eigen_" + subMeshType +"/"
        # plotting 
        U0, U0_to_W = V.sub(0).collapse()
        U1, U1_to_W = V.sub(1).collapse()

        u0_plot = fem.Function(U0)
        u1_plot = fem.Function(U1)

        u0_plot.x.array[:] = u.x.array[U0_to_W]
        u0_plot.x.scatter_forward()

        u1_plot.x.array[:] = u.x.array[U1_to_W]
        u1_plot.x.scatter_forward()   

        with io.VTXWriter(domain.comm, folder+"u0.bp", [u0_plot], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"u1.bp", [u1_plot], engine="BP4") as vtx:
            vtx.write(0.0)
        
        u0_expr_ex  = fem.Expression(u_exact[0] ,U0.element.interpolation_points())
        u0_plot_ex = fem.Function(U0)
        u0_plot_ex.interpolate(u0_expr_ex)
        u1_expr_ex  = fem.Expression(u_exact[1] ,U1.element.interpolation_points())
        u1_plot_ex = fem.Function(U1)
        u1_plot_ex.interpolate(u1_expr_ex)

        ur_expr_ex  = fem.Expression(ur_exact,U0.element.interpolation_points())
        ur_plot_ex = fem.Function(U0)
        ur_plot_ex.interpolate(ur_expr_ex)

        with io.VTXWriter(domain.comm, folder+"u0_ex.bp", [u0_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"u1_ex.bp", [u1_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        with io.VTXWriter(domain.comm, folder+"ur_ex.bp", [ur_plot_ex], engine="BP4") as vtx:
            vtx.write(0.0)
        
        # plot strain 
        V_strain = fem.FunctionSpace(domain, ("DG", k-1))
        eps_soln = epsU(u)
        folder_ecomp = folder + "strain_components/"
        strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00")
        strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10")
        strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01")
        strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11")
        eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
        strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag")


        V_stress = V_strain
        s_2x2 = sigma(eps_soln) 
        stress_dim = 3
        sigmaZZ = lam*(eps_soln[0,0] + eps_soln[1,1])
        sigma_soln = ufl.as_tensor([[s_2x2[0,0],s_2x2[0,1], 0], \
                                        [s_2x2[1,0],s_2x2[1,1], 0], \
                                        [0,0,sigmaZZ]])

        sigma_dev = sigma_soln - (1/3)*ufl.tr(sigma_soln)*ufl.Identity(stress_dim)
        VM_val = ufl.sqrt(1.5*ufl.inner(sigma_dev,sigma_dev)) 
        i += 1

