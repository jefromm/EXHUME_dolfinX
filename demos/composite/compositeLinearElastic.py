'''
Implementation of 2D Linear Elasticity equations on a composite, 
with compositon data taken from micro-ct images 
'''

from EXHUME_X.common import *
 

def epsU(u):
    return ufl.sym(ufl.grad(u))
def sigma(eps):
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(u_dim)

def interface_u(u,v,domain,dS,jump,C_u=10, eps0 = None):
    # specify normal directed away from circle 
    n = ufl.avg(weight*ufl.FacetNormal(domain))
    if eps0 == None:
        sig_u = sigma(epsU(u))
    else:
        sig_u = sigma(epsU(u)- eps0)
    sig_v = sigma(epsU(v))
    const = ufl.inner(ufl.avg(jump*u),ufl.dot(custom_avg((sig_v),E,domain),n))*dS
    adjconst = ufl.inner(ufl.avg(jump*v),ufl.dot(custom_avg((sig_u),E,domain),n))*dS
    gamma = gamma_int(C_u, E, domain)
    pen = gamma*ufl.inner(ufl.avg(jump*u),ufl.avg(jump*v))*dS
    return pen +const - adjconst

def dirichlet_u(u,v,ud,domain,ds,sgn,C_u=10):
    n = ufl.FacetNormal(domain)
    size = ufl.JacobianDeterminant(domain)
    h = size**(0.5)
    const = sgn*ufl.inner(ufl.dot(sigma(epsU(v)), n), (u-ud))*ds
    adjconst = ufl.inner(ufl.dot(sigma(epsU(u)), n), v)*ds
    gamma = C_u *E /h 
    pen = gamma*ufl.inner(v,(u-ud))*ds
    return  pen -const - adjconst 

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

if mode == 'strain':
    plane_stress = False
elif mode == 'stress':
    plane_stress = True
else: 
    print("only modes available are stress and strain")
    exit()


# guess number of bg dofs to pre allocate the matrix size 
ref = os.getcwd()[-1]
n = 10*(2**int(ref))
bg_dofs_guess = np.ceil(kHat * (2*n**2 + n*1.1))
filenames = [exOpName,exOpName]
no_fields = 2

# Domain geometry information
L = 1.6e-3

# cell markers, from mesh file
Al_ID = 1
epoxy_ID = 0 

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


sgn = 1.0 # use symmetric Nitsches 
# define displacements on top and bottom 
u_top = ufl.as_vector([-1.0e-5, -1.0e-5])
u_bottom = ufl.as_vector([0.0,0.0])

facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]

if lr >= 1: 
    # we use a for loop to define our linear algebra objects for each submesh 
    # for visualization, we also need to save each submeshes function space and material parameters 
    mesh_types = ["tri","quad"]
    Ws = []
    As =[]
    bs = []
    Ms = []
    domains = []
    us = []

    alphas = []
    lams = []
    mus = []
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

    Al_subdomain = ct.find(Al_ID)
    epoxy_subdomain  = ct.find(epoxy_ID)

    dim = domain.topology.dim
    domain.topology.create_connectivity(dim-1, dim)

    num_facets = domain.topology.index_map(dim-1).size_local
    f_to_c_conn = domain.topology.connectivity(dim-1,dim)

    num_facet_phases =len(facet_markers)
    facets = np.asarray([],dtype=np.int32)
    facets_mark = np.asarray([],dtype=np.int32)
    for phase in range(num_facet_phases):
        facets_phase = mesh.locate_entities(domain, domain.topology.dim-1, facet_functions[phase])
        facets_phase_mark = np.full_like(facets_phase, facet_markers[phase])
        facets= np.hstack((facets,facets_phase))
        facets_mark = np.hstack((facets_mark,facets_phase_mark))


    interface_facets = getInterfaceFacets(f_to_c_conn, num_facets, cell_mat)
    interface_facet_marks = interface_ID*np.ones_like(interface_facets)

    facets= np.hstack((facets,interface_facets))
    facets_mark = np.hstack((facets_mark,interface_facet_marks))
    sorted_facets = np.argsort(facets)
    ft = mesh.meshtags(domain,dim-1,facets[sorted_facets], facets_mark[sorted_facets])

    #create weight function to control integration on the interior surface 
    V_DG = fem.FunctionSpace(domain, ("DG", 0))
    weight = fem.Function(V_DG)
    weight.x.array[Al_subdomain] = 2
    weight.x.array[epoxy_subdomain] = 0
    weight.x.scatter_forward()

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


    el = ufl.FiniteElement("DG", domain.ufl_cell(),k)
    mel = ufl.MixedElement([el, el])

    W = fem.FunctionSpace(domain, mel)

    v0, v1, = ufl.TestFunction(W)
    v = ufl.as_vector([v0,v1])
    u = fem.Function(W)
    x = ufl.SpatialCoordinate(domain)
    u_dim = len(u)

    nu = fem.Function(V_DG)
    nu.x.array[Al_subdomain] = nu_Al
    nu.x.array[epoxy_subdomain] = nu_epoxy
    nu.x.scatter_forward()

    E = fem.Function(V_DG)
    E.x.array[Al_subdomain] = E_Al
    E.x.array[epoxy_subdomain] = E_epoxy
    E.x.scatter_forward()

    lam = (E*nu)/((1+nu)*(1-nu))
    mu = E/(2*(1+nu))
    if plane_stress:
        lam = 2*mu*lam/(lam+2*mu)

    # define residuals 
    res_u = ufl.inner(epsU(v),sigma(epsU(u)))*(dx(Al_ID) + dx(epoxy_ID))

    # Nitsches BC
    resD_u_t = dirichlet_u(u,v,u_top,domain,ds(top_ID),sgn)
    resD_u_b = dirichlet_u(u,v,u_bottom,domain,ds(bottom_ID),sgn)

    # Nitsches Interface 
    resI_u = interface_u(u,v,domain,dS(interface_ID),jump,C_u=100)

    res = res_u 
    res += resD_u_t 
    res += resD_u_b 
    res += resI_u 
    J = ufl.derivative(res,u)


    #dolfinx.fem.assemble_matrix(J,[])
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)
    J_petsc.assemble()
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    sizes = res_petsc.getSizes()
    M = readExOpElementwise(W,filenames,idt,domain,sizes,type =subMeshType,bg_size=no_fields*bg_dofs_guess)
    u_petsc = J_petsc.createVecLeft()
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)

    Ws += [W]
    As += [A]
    bs += [b]
    Ms += [M]
    domains +=[domain]
    us += [u]
    lams += [lam]
    mus += [mu]

if lr>= 1:
    W_tri, W_quad = Ws
    A_tri,A_quad = As
    b_tri,b_quad = bs
    M_tri,M_quad = Ms
    domain_tri,domain_quad = domains
    u_tri,u_quad = us


    # add the two matrices
    A_tri.axpy(1.0,A_quad)
    b_tri.axpy(1.0,b_quad)

    x = A_tri.createVecLeft()

    #solveKSP(A_tri,b_tri,x,monitor=False,method='gmres',rtol=1E-15, atol=1E-15)
    solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')

    transferToForeground(u_tri, x, M_tri)
    transferToForeground(u_quad, x, M_quad)

    u_solns = [u_tri,u_quad]
else: 
    A = As[0]
    b = bs[0]
    M = M[0]
    u = us[0]
    x = A.createVecLeft()
    solveKSP(A,b,x,monitor=False,method='mumps')
    transferToForeground(u, x, M)
    u_solns = [u]

if k == 2: 
    strainFileWriter = outputVTX
else:
    strainFileWriter = outputXDMF

VMs = []

i = 0 
for subMeshType in mesh_types:
    W = Ws[i] 
    u = u_solns[i] 
    lam = lams[i]
    mu = mus[i]
    domain = domains[i]

    folder = "results_" + subMeshType +"/"
    # plotting 
    U0, U0_to_W = W.sub(0).collapse()
    U1, U1_to_W = W.sub(1).collapse()

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


    s_2x2 = sigma(eps_soln) 
    stress_dim = 3
    if plane_stress:
        sigma_soln = ufl.as_tensor([[s_2x2[0,0],s_2x2[0,1], 0], \
                                    [s_2x2[1,0],s_2x2[1,1], 0], \
                                    [0,0,0]])
    else:
        sigmaZZ = lam*(eps_soln[0,0] + eps_soln[1,1])
        sigma_soln = ufl.as_tensor([[s_2x2[0,0],s_2x2[0,1], 0], \
                                    [s_2x2[1,0],s_2x2[1,1], 0], \
                                    [0,0,sigmaZZ]])
    sigma_dev = sigma_soln - (1/3)*ufl.tr(sigma_soln)*ufl.Identity(stress_dim)


    VM_val = ufl.sqrt(1.5*ufl.inner(sigma_dev ,sigma_dev )) 
    
    strainFileWriter(VM_val,V_strain,folder,"VM_stress")

    VM_expr = fem.Expression(VM_val, V_strain.element.interpolation_points())
    VM_fun= fem.Function(V_strain)
    VM_fun.interpolate(VM_expr)

    VM_max = max(VM_fun.x.array)
  
    
    VMs += [VM_max]

    i+=1 

VM_max_tri, VM_max_quad = VMs


if domain.comm.rank == 0:
    ref = os.getcwd()[-1]
    print(f"Max Von Mises stress (tri):", "{:e}".format(VM_max_tri))
    print(f"Max Von Mises stress (quad):", "{:e}".format(VM_max_quad))