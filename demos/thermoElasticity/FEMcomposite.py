'''
Implementation of 2D thermo-elasticity equations on a composite, 
with compositon data taken from micro-ct images 
Using FEM for comparison
'''


from mpi4py import MPI
from EXHUME_X.common import *
from dolfinx import mesh, fem, io, la
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

def dirichlet_T(T,q,Td,domain,ds,C_T=10):
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    n = ufl.FacetNormal(domain)
    const = (T-Td)*kappa*ufl.inner(ufl.grad(q), n)*ds
    adjconst = q*kappa*ufl.inner(ufl.grad(T), n)*ds
    gamma = C_T *kappa / h 
    pen = gamma*q*(T-Td)*ds
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
parser.add_argument('--mesh',dest='mesh',default='id_map_mesh.xdmf',
                    help='mesh file (overrides ref level)')
args = parser.parse_args()


meshFile = args.mesh

#Read in mesh
with io.XDMFFile(MPI.COMM_WORLD, meshFile, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid",ghost_mode=cpp.mesh.GhostMode.shared_facet)
    ct = xdmf.read_meshtags(domain, name="Grid")
    xdmf.close()
cell_mat = ct.values
dim = domain.topology.dim
u_dim = 2
k = 2

num_pts = domain.topology.index_map(dim-2).size_local
num_cells = domain.topology.index_map(dim).size_local

# cell markers, from mesh file
Al_ID = 0
epoxy_ID = 1 
Al_subdomain = ct.find(Al_ID)
epoxy_subdomain  = ct.find(epoxy_ID)
domain.topology.create_connectivity(dim-1, dim)
num_facets = domain.topology.index_map(dim-1).size_local
f_to_c_conn = domain.topology.connectivity(dim-1,dim)

#facet markers
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4


L = 1.6e-3
facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]

#get boundary facets with geometric function
num_facet_phases =len(facet_markers)
facets = np.asarray([],dtype=np.int32)
facets_mark = np.asarray([],dtype=np.int32)
for phase in range(num_facet_phases):
    facets_phase = mesh.locate_entities(domain, domain.topology.dim-1, facet_functions[phase])
    facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
    facets= np.hstack((facets,facets_phase))
    facets_mark = np.hstack((facets_mark,facets_phase_mark))

# mark interface ids from cell markers 
interface_facets = []
interface_facet_marks =[]
for facet in range(num_facets):
    marker = 0
    cells = f_to_c_conn.links(facet)
    for cell in cells:
        marker = marker + cell_mat[cell]
        #marker+= 1 # because the interface IDs are shifted
    if marker == 3: 
        interface_facets += [facet]
        interface_facet_marks += [interface_ID]

interface_facets_np =np.asarray(interface_facets,dtype=np.int32)
interface_facet_marks_np =np.asarray(interface_facet_marks,dtype=np.int32)

facets= np.hstack((facets,interface_facets_np))
facets_mark = np.hstack((facets_mark,interface_facet_marks_np))
sorted_facets = np.argsort(facets)
ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

# define integration measurements for the domain of interest and the interior surface of interest
dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 2*k})
ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*k})
x = ufl.SpatialCoordinate(domain)
one = 1 + x[1] - x[1]

el_u = ufl.FiniteElement("CG", domain.ufl_cell(),k)
el_t = ufl.FiniteElement("CG", domain.ufl_cell(),k-1)
el = ufl.MixedElement([el_u,el_u,el_t])
V = fem.FunctionSpace(domain, el)
no_fields = 3 

v0, v1, q  = ufl.TestFunction(V)
v = ufl.as_vector([v0,v1])
vq = ufl.as_vector([v0,v1,q])
uT = fem.Function(V)
u0,u1, T = ufl.split(uT)
u = ufl.as_vector([u0,u1])

V_DG = fem.FunctionSpace(domain, ("DG", 0))
nu_Al = 0.23
nu_epoxy = 0.358
nu = fem.Function(V_DG)
nu.x.array[Al_subdomain] = nu_Al
nu.x.array[epoxy_subdomain] = nu_epoxy
nu.x.scatter_forward()

E_Al = 320e9
E_epoxy = 3.66e9
E = fem.Function(V_DG)
E.x.array[Al_subdomain] = E_Al
E.x.array[epoxy_subdomain] = E_epoxy
E.x.scatter_forward()

lam = (E*nu)/((1+nu)*(1-nu))
mu = E/(2*(1+nu))


kappa_Al = 25.0 #W/mK
kappa_epoxy = 0.14 #W/mK 
kappa = fem.Function(V_DG)
kappa.x.array[Al_subdomain] = kappa_Al
kappa.x.array[epoxy_subdomain] = kappa_epoxy
kappa.x.scatter_forward()

alpha_Al = 15e-6
alpha_epoxy = 65e-6
alpha = fem.Function(V_DG)
alpha.x.array[Al_subdomain] = alpha_Al
alpha.x.array[epoxy_subdomain] = alpha_epoxy
alpha.x.scatter_forward()


u_top = ufl.as_vector([-1.0e-5, -1.0e-5])
u_bottom = ufl.as_vector([0.0,0.0])
T_0 = 0.0
T_top = T_0
T_bottom = 100

# define residuals 
res_T = kappa* ufl.inner(ufl.grad(q),ufl.grad(T))*(dx(Al_ID)+dx(epoxy_ID))
resD_T_t = dirichlet_T(T,q,T_top,domain,ds(top_ID))
resD_T_b = dirichlet_T(T,q,T_bottom,domain,ds(bottom_ID))

epsE = epsU(u) - epsT(T,alpha)
res_u = ufl.inner(epsU(v),sigma(epsE))*(dx(Al_ID) + dx(epoxy_ID))
resD_u_t = dirichlet_u(u,v,u_top,domain,ds(top_ID))
resD_u_b = dirichlet_u(u,v,u_bottom,domain,ds(bottom_ID))

res = res_T + resD_T_b + resD_T_t + res_u + resD_u_t + resD_u_b
J = ufl.derivative(res,uT)

# convert from dolfinx product objects to PETSc vector and matrix
res_form = fem.form(res)
res_petsc = fem.petsc.assemble_vector(res_form)
J_form = fem.form(J)
J_petsc = fem.petsc.assemble_matrix(J_form)
J_petsc.assemble()
res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

uT_soln = fem.Function(V)
uT_petsc = la.create_petsc_vector_wrap(uT_soln.x)  

# solve linear system
t_start = default_timer()
solveKSP(J_petsc,-res_petsc,uT_petsc,monitor=True,method='mumps')
uT_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
uT_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
t_stop = default_timer()
t_solve = t_stop-t_start
uT_soln.x.scatter_forward()

if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"Solver Time : {t_solve}")

if k == 2: 
    strainFileWriter = outputVTX
else:
    strainFileWriter = outputXDMF

fluxFileWriter = outputXDMF
folder = "FEMthermoElasticity/"

U0, U0_to_W = V.sub(0).collapse()
U1, U1_to_W = V.sub(1).collapse()
P, P_to_W = V.sub(2).collapse()

u0_plot = fem.Function(U0)
u1_plot = fem.Function(U1)
T_plot = fem.Function(P)

u0_plot.x.array[:] = uT_soln.x.array[U0_to_W]
u0_plot.x.scatter_forward()

u1_plot.x.array[:] = uT_soln.x.array[U1_to_W]
u1_plot.x.scatter_forward()    

T_plot.x.array[:] = uT_soln.x.array[P_to_W]
T_plot.x.scatter_forward() 

# plot displacement fields and temperature
with io.VTXWriter(domain.comm, folder+"u0.bp", [u0_plot], engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(domain.comm, folder+"u1.bp", [u1_plot], engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(domain.comm, folder+"T.bp", [T_plot], engine="BP4") as vtx:
    vtx.write(0.0)

# plot strain 
V_strain = fem.FunctionSpace(domain, ("DG", k-1))
u0_s,u1_s, T_s = ufl.split(uT_soln)
u_s = ufl.as_vector([u0_s,u1_s])
eps_soln = epsU(u_s) - epsT(T_s,alpha)
folder_ecomp = folder + "strain_components/"
strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00")
strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10")
strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01")
strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11")
eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag")
   
# plot temperature gradient 
V_flux = fem.FunctionSpace(domain, ("DG", k-2))
q_mag_val = ufl.sqrt(ufl.dot(ufl.grad(T_s),ufl.grad(T_s)))
fluxFileWriter(q_mag_val,V_flux,folder,"q_mag")
q_flux = ufl.grad(T_s)
fluxFileWriter(q_flux[0],V_flux,folder,"q_0")
fluxFileWriter(q_flux[1],V_flux,folder,"q_1")


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
strainFileWriter(VM_val,V_strain,folder,"VM_stress")
