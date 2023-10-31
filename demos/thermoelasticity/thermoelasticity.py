'''
Implementation of 2D thermoelasticity problem, using mixed elements 
Based on the section 5 of https://link.springer.com/article/10.1007/s00466-023-02306-x
'''
from EXHUME_X.common import *

def epsU(u):
    return ufl.sym(ufl.grad(u))
def epsT(T,alpha):
    return alpha*(T - T_0)*ufl.Identity(u_dim)
def sigma(eps):
    return 2.0*mu*eps+ lam*ufl.tr(eps)*ufl.Identity(u_dim)

#prescribe the exact solution at the boundaries 
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

def interface_T(T,q,domain,dS,jump,C_T=10):
    # specify normal directed away from circle 
    n = ufl.avg(w_inside*ufl.FacetNormal(domain))
    const = ufl.avg(jump*T) * ufl.dot(custom_avg((kappa*ufl.grad(q)),kappa,domain),n)*dS
    adjconst = ufl.avg(jump*q) * ufl.dot(custom_avg((kappa*ufl.grad(T)),kappa,domain),n)*dS
    gamma = gamma_int(C_T, kappa, domain)
    pen = gamma*ufl.avg(jump*T)*ufl.avg(jump*q)*dS
    return const - adjconst + pen 

def interface_u(u,v,domain,dS,jump,C_u=10):
    # specify normal directed away from circle 
    n = ufl.avg(w_inside*ufl.FacetNormal(domain))
    sig_u = sigma(epsU(u))
    sig_v = sigma(epsU(v))
    const = ufl.inner(ufl.avg(jump*u),ufl.dot(custom_avg((sig_v),E,domain),n))*dS
    adjconst = ufl.inner(ufl.avg(jump*v),ufl.dot(custom_avg((sig_u),E,domain),n))*dS
    gamma = gamma_int(C_u, E, domain)
    pen = gamma*ufl.inner(ufl.avg(jump*u),ufl.avg(jump*v))*dS
    return const - adjconst + pen 

def Left(x):
    return np.isclose(x[0], 0)
def Right(x):
    return np.isclose(x[0], 2*L)
def Top(x):
    return np.isclose(x[1], L)
def Bottom(x):
    return np.isclose(x[1], 0)


parser = argparse.ArgumentParser()
parser.add_argument('--ku',dest='ku',default=1,
                    help='FG displacement polynomial degree.')
parser.add_argument('--kHatu',dest='kHatu',default=None,
                    help='Background displacement spline polynomial degree, default to FG degree ku')
parser.add_argument('--uSpline',dest='uSpline',default=None,
                    help='Background displacement spline T-matrices file')
parser.add_argument('--kT',dest='kT',default=1,
                    help='FG displacement polynomial degree.')
parser.add_argument('--kHatT',dest='kHatT',default=None,
                    help='Background temperature spline polynomial degree, default to FG degree ku')
parser.add_argument('--TSpline',dest='TSpline',default=None,
                    help='Background temperature spline T-matrices file')
parser.add_argument('--lr',dest='lr',default=0,
                    help='Level of local refinement, for data reporting, default 0')
parser.add_argument('--mode',dest='mode',default='strain',
                    help='strain or stress, refering to plane strain or plane stress (default is strain)')
args = parser.parse_args()


ku = int(args.ku)
kHatu=args.kHatu
if kHatu is None:
    kHatu = ku 
else:
    kHatu = int(args.kHatu)
kT = int(args.kT)
kHatT=args.kHatT
if kHatT is None:
    kHatT = kT 
else:
    kHatT = int(args.kHatT)

uSpline=args.uSpline
TSpline=args.TSpline

lr = int(args.lr)
mode = args.mode
if mode == 'strain':
    plane_stress = False
elif mode == 'stress':
    plane_stress = True
else: 
    print("only modes available are stress and strain")
    exit()

# guess number of bg dofs to pre allocate the matrix size 
ref = os.getcwd()[-1]
n = 8*(2**int(ref))
bg_dofs_guess = 2*np.ceil(kHatu * (2*n**2 + n*1.1)) + np.ceil(kHatT * (2*n**2 + n*1.1))

if uSpline == None:
    if kHatu == 1:
        exOpUName = 'Elemental_Extraction_Operators_B0.hdf5'
    else:
        exOpUName = 'Elemental_Extraction_Operators_B1.hdf5'
else: 
    exOpUName = uSpline
if TSpline == None:
    if kHatT == 1:
        exOpTName = 'Elemental_Extraction_Operators_B0.hdf5'
    else:
        exOpTName = 'Elemental_Extraction_Operators_B1.hdf5'
else: 
    exOpTName = TSpline

filenames = [exOpUName,exOpUName,exOpTName]

# Domain geometry information
L = 2.0
R = 0.75
centerx = 2.0
centery = 1.0

# constant heat body load in inclusion only 
q_right_const = 100.0  # W/m^2 or W/m^3 
u_left = ufl.as_vector([0,0])
T_0 = 0.0
T_left = T_0
nu = 0.3
u_dim = 2

#material properties 
kappa_inside = 1.0 #W/mK
kappa_outside = 1.0 #W/mK 
alpha_inside = 1e-4
alpha_outside = 1e-5
E_inside = 1.0
E_outside = 1.0

# cell markers, from mesh file
inside_ID = 0
outside_ID = 1

# facet markers, user specified
top_ID = 0 
bottom_ID = 1
left_ID = 2
right_ID = 3
interface_ID = 4

facet_markers = [top_ID, bottom_ID, left_ID, right_ID]
facet_functions = [Top, Bottom, Left, Right]
num_facet_phases =len(facet_markers)

if lr >= 1: 
    # we use a for loop to define our linear algebra objects for each submesh 
    # for visualization, we also need to save each submeshes function space and material parameters 
    mesh_types = ["tri","quad"]
    Vs = []
    As =[]
    bs = []
    Ms = []
    uTs =[]
    us = []
    Ts = []
    alphas = []
    lams = []
    mus = []
    domains = []
else: 
    # no local refinement, assume single tri mesh 
    mesh_types = ["tri"]

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
    interface_facets = getInterfaceFacets(f_to_c_conn, num_facets, cell_mat)
    interface_facet_marks = interface_ID*np.ones_like(interface_facets)
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

    # create DG function to calculate the jump 
    # Using the notation from (Schmidt 2023), the interior is material m and the exterior is n 
    # jump == [[.]] = (.)^m - (.)^n 
    jump = fem.Function(V_DG)
    jump.x.array[inside_subdomain] = 2
    jump.x.array[outside_subdomain] = -2
    jump.x.scatter_forward()

    # define integration measurements for the domain of interest and the interior surface of interest
    dx = ufl.Measure('dx',domain=domain,subdomain_data=ct,metadata={'quadrature_degree': 2*ku})
    ds = ufl.Measure("ds",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*ku})
    ds_exterior = ds(right_ID) + ds(left_ID) + ds(top_ID) + ds(bottom_ID)
    dS = ufl.Measure("dS",domain=domain,subdomain_data=ft,metadata={'quadrature_degree': 2*ku})

    el_u = ufl.FiniteElement("DG", domain.ufl_cell(),ku)
    el_T = ufl.FiniteElement("DG", domain.ufl_cell(),kT)
    mel = ufl.MixedElement([el_u, el_u, el_T])

    V = fem.FunctionSpace(domain, mel)
    no_fields = 3 

    v0, v1, q  = ufl.TestFunction(V)
    v = ufl.as_vector([v0,v1])
    vq = ufl.as_vector([v0,v1,q])
    uT = fem.Function(V)
    u0,u1, T = ufl.split(uT)
    u = ufl.as_vector([u0,u1])

    # define material properties
    kappa = fem.Function(V_DG)
    kappa.x.array[inside_subdomain] = kappa_inside
    kappa.x.array[outside_subdomain] = kappa_outside
    kappa.x.scatter_forward()

    E = fem.Function(V_DG)
    E.x.array[inside_subdomain] = E_inside
    E.x.array[outside_subdomain] = E_outside
    E.x.scatter_forward()
    
    alpha = fem.Function(V_DG)
    alpha.x.array[inside_subdomain] = alpha_inside
    alpha.x.array[outside_subdomain] = alpha_outside
    alpha.x.scatter_forward()
    
    lam = E*nu /((1 + nu)*(1 - 2*nu))
    mu = (E)/(2*(1+nu))
    if plane_stress:
        lam = 2*mu*lam/(lam+2*mu)

    # define heat flux
    x = ufl.SpatialCoordinate(domain)
    q_right = q_right_const*ufl.sin(10.0*x[1])+110.0


    # define residuals 
    res_T = kappa* ufl.inner(ufl.grad(q),ufl.grad(T))*( dx(inside_ID)+dx(outside_ID))\
          - ufl.inner(q,q_right)*((ds(right_ID) ))    
    epsE = epsU(u) - epsT(T,alpha)
    res_u = ufl.inner(epsU(v),sigma(epsE))*(dx(inside_ID) + dx(outside_ID))
    resD_u_l = dirichlet_u(u,v,u_left,domain,ds(left_ID))
    resD_T_l = dirichlet_T(T,q,T_left,domain,ds(left_ID))

    resI_T = interface_T(T,q,domain,dS(interface_ID),jump,C_T=100)
    resI_u = interface_u(u,v,domain,dS(interface_ID),jump,C_u=100)

    res = res_u 
    res += res_T 
    res += resI_T 
    res += resI_u 
    res += resD_u_l 
    res += resD_T_l 

    # use automatic differentiation to compute the Gateux derivative 
    J = ufl.derivative(res,uT)
    res_form = fem.form(res)
    res_petsc = fem.petsc.assemble_vector(res_form)
    J_form = fem.form(J)
    J_petsc = fem.petsc.assemble_matrix(J_form)
    J_petsc.assemble()
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    sizes = res_petsc.getSizes()

    #read in mesh extraction operator 
    M = readExOpElementwise(V,filenames,idt,domain,sizes,type=subMeshType,bg_size=bg_dofs_guess)
    u_petsc = J_petsc.createVecLeft()

    # assemble linear system 
    A,b = assembleLinearSystemBackground(J_petsc,-res_petsc,M)

    # save submesh parameters for plotting 
    Vs +=[V]
    As += [A]
    bs += [b]
    Ms += [M]
    uTs += [uT]
    us += [u]
    Ts +=[T]
    alphas += [alpha]
    lams += [lam]
    mus += [mu]
    domains += [domain]

if lr >= 1:
    # access submesh quantities 
    A_tri,A_quad = As
    b_tri,b_quad = bs
    M_tri,M_quad = Ms
    uT_tri,uT_quad = uTs

    # add the two matrices
    A_tri.axpy(1.0,A_quad)
    b_tri.axpy(1.0,b_quad)
    x = A_tri.createVecLeft()


    #solveKSP(A_tri,b_tri,x,monitor=False,method='gmres',rtol=1E-15, atol=1E-15)
    solveKSP(A_tri,b_tri,x,monitor=False,method='mumps')

    transferToForeground(uT_tri, x, M_tri)
    transferToForeground(uT_quad, x, M_quad)
else:
    A = As[0]
    b = bs[0]
    M = M[0]
    uT = uTs[0]

    x = A.createVecLeft()
    solveKSP(A,b,x,monitor=False,method='mumps')
    transferToForeground(uT, x, M)



# generate files for visualizing results 
if ku == 2: 
    strainFileWriter = outputVTX
else:
    strainFileWriter = outputXDMF
if kT == 2: 
    fluxFileWriter = outputVTX
else:
    fluxFileWriter = outputXDMF

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

    folder =  subMeshType +"/"

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

    # plot solutions 
    with io.VTXWriter(domain.comm, folder+"u0.bp", [u0_plot], engine="BP4") as vtx:
        vtx.write(0.0)
    with io.VTXWriter(domain.comm, folder+"u1.bp", [u1_plot], engine="BP4") as vtx:
        vtx.write(0.0)
    with io.VTXWriter(domain.comm, folder+"T.bp", [T_plot], engine="BP4") as vtx:
        vtx.write(0.0)

    # plot strain 
    V_strain = fem.FunctionSpace(domain, ("DG", ku-1))
    eps_soln = epsU(u)
    folder_ecomp = folder + "strain_components/"
    strainFileWriter(eps_soln[0,0],V_strain,folder_ecomp,"e00")
    strainFileWriter(eps_soln[1,0],V_strain,folder_ecomp,"e10")
    strainFileWriter(eps_soln[0,1],V_strain,folder_ecomp,"e01")
    strainFileWriter(eps_soln[1,1],V_strain,folder_ecomp,"e11")
    eps_sol_mag = ufl.sqrt(ufl.inner(eps_soln,eps_soln)) 
    strainFileWriter(eps_sol_mag,V_strain,folder,"eps_mag")
    
    #plot heat flux    
    q_mag_val = ufl.sqrt(ufl.dot(ufl.grad(T),ufl.grad(T)))
    fluxFileWriter(q_mag_val,V_strain,folder,"q_mag")
    

    # plot von Mises stress
    eps_soln = epsU(u) -epsT(T,alpha)
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
    i+=1