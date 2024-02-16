"""
Simple poisson problem using the method of manufactured solution
adapted from Jorgen Dokken's dolfinx tutorial: 
https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html
"""

from EXHUME_X.common import *


# define simple mesh 
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# define a function space with the mesh discretization
V = fem.FunctionSpace(domain, ("CG", 1))

# create a funtion to use as our source term
def u_exact_fun(x): 
    return  np.sin(x[1] + x[0]+ 0.1)*np.cos(x[1] - x[0]- 0.1)
# as a function for visualization
u_ex_disp = fem.Function(V)
u_ex_disp.interpolate(u_exact_fun)

# as a ufl object for the body force and BC
def u_exact_ufl(x): 
    return ufl.sin(x[1] + x[0]+ 0.1)*ufl.cos(x[1] - x[0]- 0.1)
x = ufl.SpatialCoordinate(domain)
u_ex = u_exact_ufl(x) 
f = -ufl.div(ufl.grad(u_ex))

# define trial and test functions 
u = fem.Function(V)
v = ufl.TestFunction(V)

# define a source term
f = -ufl.div(ufl.grad(u_ex))

# define the interior residual
res = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx\
    - ufl.inner(f,v)* ufl.dx

# use Nitsche's method to weakly enforce Dirichlet boundary condtions 
n = ufl.FacetNormal(domain)
size = ufl.JacobianDeterminant(domain)
h_E = size**(0.5)
beta = fem.Constant(domain, default_scalar_type(10))
const = ufl.inner(u_ex-u, ufl.dot(ufl.grad(v), n))*ufl.ds
adjconst = -ufl.inner(ufl.dot(ufl.grad(u), n), v)*ufl.ds 
penalty = beta*h_E**(-1)*ufl.inner(u-u_ex, v)*ufl.ds
res_D = const + adjconst + penalty 

res += res_D

# take Jacobian using automatic differentiation
J = ufl.derivative(res,u)

# assemble forms into PETSc objects
res_petsc = fem.petsc.assemble_vector(fem.form(res))
J_petsc = fem.petsc.assemble_matrix(fem.form(J))
res_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
res_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

u_petsc = la.create_petsc_vector_wrap(u.x)

# use custom KSP solver to solve linear system 
solveKSP(J_petsc,-res_petsc,u_petsc,method='fgmres',PC='jacobi')
u.x.scatter_forward()

# compute error
folder = 'result/'
L2_error = fem.form(ufl.inner(u - u_ex, u - u_ex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
H10_error = fem.form(ufl.inner(ufl.grad(u - u_ex), ufl.grad(u - u_ex)) *ufl.dx)
error_local = fem.assemble_scalar(H10_error)
error_H10 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# Output errors
if domain.comm.rank == 0:
    print(f"L2 Error): {error_L2}")
    print(f"H10 Error: {error_H10}")


# output solution and exact function for visualization

with io.VTXWriter(domain.comm, folder +"u.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(domain.comm, folder +"u_ex.bp", [u_ex_disp], engine="BP4") as vtx:
    vtx.write(0.0)