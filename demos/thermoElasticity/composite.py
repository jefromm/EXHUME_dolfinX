'''
Implementation of 2D thermo-elasticity equations on a composite, 
with compositon data taken from micro-ct images 
'''


from mpi4py import MPI
from EXHUME_X.common import *
from dolfinx import mesh, fem, io
import os
import ufl
from petsc4py import PETSc
import numpy as np 
from timeit import default_timer

def epsU(u):
    '''
    Total Strain
    '''
    return ufl.sym(ufl.grad(u))
def epsT(T,alpha):
    '''
    Thermal strain
    '''
    return alpha*(T - T_0)*ufl.Identity(u_dim)
def sigma(eps):
    '''
    Stress 
    '''
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(u_dim)

def interface_u(u,v,domain,dS,jump,C_u=10):
    n = ufl.avg(w_inside*ufl.FacetNormal(domain))
    sig_u = sigma(epsU(u))
    sig_v = sigma(epsU(v))
    const = ufl.inner(ufl.avg(jump*u),ufl.dot(custom_avg((sig_v),E,domain),n))*dS
    adjconst = ufl.inner(ufl.avg(jump*v),ufl.dot(custom_avg((sig_u),E,domain),n))*dS
    gamma = gamma_int(C_u, E, domain)
    pen = gamma*ufl.inner(ufl.avg(jump*u),ufl.avg(jump*v))*dS
    return const - adjconst + pen 


def dirichlet_u(u,v,ud,domain,ds,C_u=10):
    n = ufl.FacetNormal(domain)
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    const = ufl.inner(ufl.dot(sigma(epsU(v)), n), (u-ud))*ds
    adjconst = ufl.inner(ufl.dot(sigma(epsU(u)), n), v)*ds
    gamma = C_u *E /h 
    pen = gamma*ufl.inner(v,(u-ud))*ds
    return const - adjconst + pen 


def interface_T(T,q,domain,dS,jump,C_T=10):
    n = ufl.avg(w_inside*ufl.FacetNormal(domain))
    const = ufl.avg(jump*T) * ufl.dot(custom_avg((kappa*ufl.grad(q)),kappa,domain),n)*dS
    adjconst = ufl.avg(jump*q) * ufl.dot(custom_avg((kappa*ufl.grad(T)),kappa,domain),n)*dS
    gamma = gamma_int(C_T, kappa, domain)
    pen = gamma*ufl.avg(jump*T)*ufl.avg(jump*q)*dS
    return const - adjconst + pen 

def dirichlet_T(T,q,Td,domain,ds,C_T=10):
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    n = ufl.FacetNormal(domain)
    const = (T-Td)*kappa*ufl.inner(ufl.grad(q), n)*ds
    adjconst = q*kappa*ufl.inner(ufl.grad(T), n)*ds
    gamma = C_T *kappa / h 
    pen = gamma*q*(T-Td)*ds
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sym',dest='symmetric',default=True,
                    help='True for symmetric Nitsche; False for nonsymmetric')                   
parser.add_argument('--lr',dest='lr',default=0,
                    help='level of local refinement, for data reporting, default 0')
args = parser.parse_args()

k=2
lr = int(args.lr)
ref = 3 + int(lr)
n = 10*(2**int(ref))
bg_dofs_guess = 2*np.ceil(k * (2*n**2 + n*2.1))
exOpName_u = 'Elemental_Extraction_Operators_B1.hdf5'
exOpName_T = 'Elemental_Extraction_Operators_B0.hdf5'
filenames = [exOpName_u, exOpName_u, exOpName_T]
u_dim = 2
L = 1.6e-3

# cell markers, from mesh file
Al_ID = 0
epoxy_ID = 1 

#facet markers
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4

# material properties
nu_Al = 0.23
nu_epoxy = 0.358
E_Al = 320e9
E_epoxy = 3.66e9
kappa_Al = 25.0
kappa_epoxy = 0.14
alpha_Al = 15e-6
alpha_epoxy = 65e-6

# boundary conditions
u_top = ufl.as_vector([-1.0e-5, -1.0e-5])
u_bottom = ufl.as_vector([0.0,0.0])
T_0 = 0.0
T_top = T_0
T_bottom = 100
no_fields = 3


facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]

mesh_types = ["tri","quad"]
Vs = []
As =[]
bs = []
Ms = []
domains = []
us = []
Ts = []
uTs =[]
t_exs = []
dxs = []
alphas = []
lams = []
mus = []

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
    Al_subdomain = ct.find(Al_ID)
    epoxy_subdomain  = ct.find(epoxy_ID)
    dim = domain.topology.dim
    domain.topology.create_connectivity(dim-1, dim)

    num_facets = domain.topology.index_map(dim-1).size_local
    f_to_c_conn = domain.topology.connectivity(dim-1,dim)

    #get boundary facets with geometric function
    num_facet_phases =len(facet_markers)
    facets = np.asarray([],dtype=np.int32)
    facets_mark = np.asarray([],dtype=np.int32)
    for phase in range(num_facet_phases):
        facets_phase = mesh.locate_entities(domain, domain.topology.dim-1, facet_functions[phase])
        facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
        facets= np.hstack((facets,facets_phase))
        facets_mark = np.hstack((facets_mark,facets_phase_mark))
    
    # determine interface facets from cell markers
    interface_facets = []
    interface_facet_marks =[]
    for facet in range(num_facets):
        marker = 0
        cells = f_to_c_conn.links(facet)
        for cell in cells:
            marker = marker + cell_mat[cell]
            marker+= 1 # because the interface IDs are shifted
        if marker == 3: 
            interface_facets += [facet]
            interface_facet_marks += [interface_ID]


    interface_facets_np =np.asarray(interface_facets,dtype=np.int32)
    interface_facet_marks_np =np.asarray(interface_facet_marks,dtype=np.int32)

    facets= np.hstack((facets,interface_facets_np))
    facets_mark = np.hstack((facets_mark,interface_facet_marks_np))
    sorted_facets = np.argsort(facets)
    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])

    # create DG function to control integration on interior cells
    V_DG = fem.FunctionSpace(domain, ("DG", 0))
    w_inside = fem.Function(V_DG)
    w_inside.x.array[Al_subdomain] = 2
    w_inside.x.array[epoxy_subdomain] = 0
    w_inside.x.scatter_forward()
    w_outside = fem.Function(V_DG)
    w_outside.x.array[Al_subdomain] = 0
    w_outside.x.array[epoxy_subdomain] = 2
    w_outside.x.scatter_forward()

    # create DG function to calculate the jump 
    # Using the notation from (Schmidt 2023), the interior is material m and the exterior is n 
    # jump == [[.]] = (.)^m - (.)^n 
    jump = fem.Function(V_DG)
    jump.x.array[Al_subdomain] = 2
    jump.x.array[epoxy_subdomain] = -2
    jump.x.scatter_forward()
    domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

    # create DG function to calculate the jump 
    # Using the notation from (Schmidt 2023), the interior is material m and the exterior is n 
    # jump == [[.]] = (.)^m - (.)^n 
    jump = fem.Function(V_DG)
    jump.x.array[Al_subdomain] = 2
    jump.x.array[epoxy_subdomain] = -2
    jump.x.scatter_forward()

    # define integration measurements for the domain of interest and the interior surface of interest
    dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 2*k})
    ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})
    ds_exterior = ds(right_ID) + ds(left_ID) + ds(top_ID) + ds(bottom_ID)
    dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})


    el_u = ufl.FiniteElement("DG", domain.ufl_cell(),2)
    el_T = ufl.FiniteElement("DG", domain.ufl_cell(),2)
    mel = ufl.MixedElement([el_u, el_u, el_T])

    V = fem.FunctionSpace(domain, mel)
    v0, v1, q  = ufl.TestFunction(V)
    v = ufl.as_vector([v0,v1])
    vq = ufl.as_vector([v0,v1,q])
    uT = fem.Function(V)
    u0,u1, T = ufl.split(uT)
    u = ufl.as_vector([u0,u1])

    nu = fem.Function(V_DG)
    nu.x.array[Al_subdomain] = nu_Al
    nu.x.array[epoxy_subdomain] = nu_epoxy
    nu.x.scatter_forward()

    E = fem.Function(V_DG)
    E.x.array[Al_subdomain] = E_Al
    E.x.array[epoxy_subdomain] = E_epoxy
    E.x.scatter_forward()

    kappa = fem.Function(V_DG)
    kappa.x.array[Al_subdomain] = kappa_Al
    kappa.x.array[epoxy_subdomain] = kappa_epoxy
    kappa.x.scatter_forward()

    alpha = fem.Function(V_DG)
    alpha.x.array[Al_subdomain] = alpha_Al
    alpha.x.array[epoxy_subdomain] = alpha_epoxy
    alpha.x.scatter_forward()

    inside = fem.Function(V_DG)
    inside.x.array[Al_subdomain] = 1.0
    inside.x.array[epoxy_subdomain] = 0.0
    inside.x.scatter_forward()

    lam = (E*nu)/((1+nu)*(1-nu))
    mu = E/(2*(1+nu))

    # define residuals
    res_T = kappa* ufl.inner(ufl.grad(q),ufl.grad(T))*(dx(Al_ID)+dx(epoxy_ID))
    resD_T_t = dirichlet_T(T,q,T_top,domain,ds(top_ID))
    resD_T_b = dirichlet_T(T,q,T_bottom,domain,ds(bottom_ID))

    epsE = epsU(u) - epsT(T,alpha)
    res_u = ufl.inner(epsU(v),sigma(epsE))*(dx(Al_ID) + dx(epoxy_ID))
    resD_u_t = dirichlet_u(u,v,u_top,domain,ds(top_ID))
    resD_u_b = dirichlet_u(u,v,u_bottom,domain,ds(bottom_ID))
    
    resI_T = interface_T(T,q,domain,dS(interface_ID),jump,C_T=100)
    resI_u = interface_u(u,v,domain,dS(interface_ID),jump,C_u=100)

    res = res_T + resD_T_b + resD_T_t + res_u + resD_u_t + resD_u_b + resI_T + resI_u

    # take Jacobian 
    J = ufl.derivative(res,uT)
    
    # convert from dolfinx product objects to PETSc vector and matrix
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
    bg_guess = no_fields*bg_dofs_guess
    t_start = default_timer()
    M = readExOpElementwise(V,filenames,idt,domain,sizes,type =subMeshType,bg_size=no_fields*bg_dofs_guess)
    t_stop = default_timer()
    t_ex = t_stop-t_start

    # assemble Background linear system 
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)

    # save quantities needed for both submesh types
    Vs +=[V]
    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    us += [u]
    Ts += [T]
    uTs += [uT]
    dxs += [dx] 
    t_exs += [t_ex]
    alphas += [alpha]
    lams += [lam]
    mus += [mu]

A_tri,A_quad = As
b_tri,b_quad = bs
M_tri,M_quad = Ms
domain_tri,domain_quad = domains
uT_tri,uT_quad = uTs
u_tri,u_quad = us
T_tri,T_quad = Ts
dx_tri,dx_quad = dxs
t_ex_tri,t_ex_quad = t_exs

# add the two matrices
A_tri.axpy(1.0,A_quad)
b_tri.axpy(1.0,b_quad)

x = A_tri.createVecLeft()
# solve background linear system 
t_start = default_timer()
solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')
t_stop = default_timer()
t_solve = t_stop-t_start

# transfer to foreground 
transferToForeground(uT_tri, x, M_tri)
transferToForeground(uT_quad, x, M_quad)

folder = "thermoelasticity_w_q/"

if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"Extraction Time (tris): {t_ex_tri}")
    print(f"Extraction Time (quads): {t_ex_quad}")
    print(f"Solver Time : {t_solve}")


strainFileWriter = outputVTX
fluxFileWriter = strainFileWriter
i = 0 
for subMeshType in mesh_types:
    W = Vs[i] 
    uT = uTs[i]
    u = us[i]
    T = Ts[i]
    alpha = alphas[i]
    lam = lams[i]
    mu = mus[i]
    domain = domains[i]

    U0, U0_to_W = W.sub(0).collapse()
    U1, U1_to_W = W.sub(1).collapse()
    P, P_to_W = W.sub(2).collapse()

    u0_plot = fem.Function(U0)
    u1_plot = fem.Function(U1)
    T_plot = fem.Function(P)

    u0_plot.x.array[:] = uT.x.array[U0_to_W]
    u0_plot.x.scatter_forward()

    u1_plot.x.array[:] = uT.x.array[U1_to_W]
    u1_plot.x.scatter_forward()    

    T_plot.x.array[:] = uT.x.array[P_to_W]
    T_plot.x.scatter_forward() 

    # plot displacement fields and temperature
    with io.VTXWriter(domain.comm, folder + "u0_"+ subMeshType+ ".bp", [u0_plot], engine="BP4") as vtx:
        vtx.write(0.0)
    with io.VTXWriter(domain.comm, folder + "u1_"+ subMeshType+ ".bp", [u1_plot], engine="BP4") as vtx:
        vtx.write(0.0)
    with io.VTXWriter(domain.comm, folder + "T_"+ subMeshType+ ".bp", [T_plot], engine="BP4") as vtx:
        vtx.write(0.0)

    u_mag = ufl.sqrt(ufl.inner(u,u))
    outputVTX(u_mag,U0,folder,"u_mag"+subMeshType)

    # plot strain 
    V_strain = fem.FunctionSpace(domain, ("DG", k-1))
    eps_soln = epsU(u) - epsT(T,alpha)
    folder_ecomp = folder + "strain_components/"
    strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00"+subMeshType)
    strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10"+subMeshType)
    strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01"+subMeshType)
    strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11"+subMeshType)
    eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
    strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag"+subMeshType)
    
    # plot temperature gradient 
    V_flux = fem.FunctionSpace(domain, ("DG", k-1))
    q_mag_val = ufl.sqrt(ufl.dot(ufl.grad(T),ufl.grad(T)))
    q_flux = ufl.grad(T)
    fluxFileWriter(q_mag_val,V_flux,folder,"q_mag"+subMeshType)
    fluxFileWriter(q_flux[0],V_flux,folder,"q_0"+subMeshType)
    fluxFileWriter(q_flux[1],V_flux,folder,"q_1"+subMeshType)

    # plot stress
    s_2x2 = sigma(eps_soln) 
    stress_dim = 3
    #note- assumes plain strain 
    sigmaZZ = lam*(eps_soln[0,0] + eps_soln[1,1])
    sigma_soln = ufl.as_tensor([[s_2x2[0,0],s_2x2[0,1], 0], \
                                    [s_2x2[1,0],s_2x2[1,1], 0], \
                                    [0,0,sigmaZZ]])
    sigma_dev = sigma_soln - (1/3)*ufl.tr(sigma_soln)*ufl.Identity(stress_dim)
    VM_val = ufl.sqrt(1.5*ufl.inner(sigma_dev ,sigma_dev )) 
    strainFileWriter(VM_val,V_strain,folder,"VM_stress"+subMeshType)
    i+=1