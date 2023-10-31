'''
Workflow to convert the mesh files generated by XTK from Exodus to XDMF format

run with: 

python3 mesh_convert.py --fi "Foreground_0.exo" --fo "foreground_0.xdmf"
'''
import meshio    
import numpy as np 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fi',dest='fi',default="foreground_mesh.exo",
                    help='Input mesh file')
parser.add_argument('--fo',dest='fo',default="meshes",
                    help='Output mesh folder')

                    
args = parser.parse_args()
FILE_IN = args.fi
FILE_OUT = args.fo

def makeIDsConsecutive(original_cells, original_points):
    #first create maps from the old ID to new, and vis versa 
    cons2noncons = np.unique(original_cells.flatten()) #index is new ID, value is old
    noncons2cons = np.zeros(np.max(cons2noncons)+1,dtype=np.int32) - 1 # index is old ID, value is new 
    i = 0
    for nonconsID in cons2noncons:
        noncons2cons[nonconsID] = i # index is old ID, value is new 
        new_pt = original_points[nonconsID].reshape(1,3)
        if i == 0:
            new_points = new_pt
        else:
            new_points = np.concatenate((new_points, new_pt), axis=0)
        i = i +1


    #manipulate array format
    row, col = np.shape(original_cells)
    cells_flat = original_cells.flatten()
    new_cells = np.zeros_like(cells_flat) - 1
    i = 0
    for ID in cells_flat:
        new_cells[i] = noncons2cons[ID]
        i = i + 1
    new_cells = new_cells.reshape(row, col)
    return new_cells, new_points, noncons2cons

def trimHOPoints(original_cells, original_points):
    used_IDS = np.unique(original_cells.flatten()) #index is new ID, value is old
    i = 0 
    for used_ID in used_IDS:
        
        if used_ID != i:
            print('This is the problem!!')
            exit()
        new_pt = original_points[used_ID].reshape(1,3)
        if i == 0:
            new_points = new_pt
        else:
            new_points = np.concatenate((new_points, new_pt), axis=0)
        i = i +1
    return new_points


print(">>> Reading the mesh file...")

PATH = ""
filename = PATH + FILE_IN
exo = meshio.read(filename)

print(">>> Creating material data ...")
IDs = exo.cell_data 

tri_cells = []
quad_cells = []
block = 0 
t_block = 0
q_block = 0 
for cell in exo.cells:
    if cell.type == "triangle6" or cell.type == "triangle":
        t_cell_type= cell.type
        node_list = cell.data
        cell_type = cell.type
        cell_IDs = IDs['cell_id'][block]
        if t_block == 0:
            t_cells = node_list
            t_xdmf_IDs = cell_IDs
            t_materials = np.zeros((1,len(node_list)))
            t_num_block_cells = len(node_list)
        else:
            t_cells = np.concatenate((t_cells, node_list))
            t_xdmf_IDs = np.concatenate((t_xdmf_IDs, cell_IDs))
            t_materials = np.concatenate((t_materials, t_block*np.ones((1,len(node_list)))),axis = 1)
            t_num_block_cells += len(node_list)

        block += 1
        t_block += 1
    else: 
        q_cell_type = cell_type
        node_list = cell.data
        cell_type = cell.type
        cell_IDs = IDs['cell_id'][block]
        if q_block == 0:
            q_cells = node_list
            q_xdmf_IDs = cell_IDs
            q_materials = np.zeros((1,len(node_list)))
            q_num_block_cells = len(node_list)
            
        else:
            q_cells = np.concatenate((q_cells, node_list))
            q_xdmf_IDs = np.concatenate((q_xdmf_IDs, cell_IDs))
            q_materials = np.concatenate((q_materials, q_block*np.ones((1,len(node_list)))),axis = 1)
            q_num_block_cells += len(node_list)
        block += 1
        q_block += 1

points = exo.points

if t_cell_type == 'triangle6':
    t_cells = np.delete(t_cells, [3, 4, 5], 1)
    t_cell_type = "triangle"

t_cells, t_points, HO_nodeIDmap = makeIDsConsecutive(t_cells, points)
# get rid of all points that are not in this mesh 
t_points = trimHOPoints(t_cells,t_points)

# delete z-coordinate 
t_points = np.delete(t_points, 2, 1)

t_ID_map = np.asarray([t_xdmf_IDs])



print(">>> Creating tri mesh ...")

mesh = meshio.Mesh(points=t_points,cells=[(t_cell_type, t_cells)],
    cell_data={"ID_map":t_ID_map})
t_file =FILE_OUT + "/tri.xdmf"
meshio.write(t_file,mesh)

# export a second mesh with material information
mesh = meshio.Mesh(points=t_points,cells=[(t_cell_type, t_cells)],
    cell_data={"material":t_materials,})
t_m_file = FILE_OUT + "/tri_materials.xdmf"
meshio.write(t_m_file ,mesh)

if q_cell_type == 'quad9':
    q_cells = np.delete(q_cells, [4, 5, 6, 7, 8], 1)
    q_cell_type = "quad"

# make quad meshes 
q_cells,q_points, HO_nodeIDmap = makeIDsConsecutive(q_cells, points)

# get rid of all points that are not in this mesh 
q_points = trimHOPoints(q_cells,q_points)

# delete z-coordinate 
q_points = np.delete(q_points, 2, 1)
q_ID_map = np.asarray([q_xdmf_IDs])

print(">>> Creating quad mesh ...")
mesh = meshio.Mesh(points=q_points,cells=[(q_cell_type, q_cells)],
    cell_data={"ID_map":q_ID_map})
q_file = FILE_OUT + "/quad.xdmf"
meshio.write(q_file,mesh)

mesh = meshio.Mesh(points=q_points,cells=[(q_cell_type, q_cells)],
    cell_data={"material":q_materials,})
q_m_file = FILE_OUT + "/quad_materials.xdmf"
meshio.write(q_m_file ,mesh)
