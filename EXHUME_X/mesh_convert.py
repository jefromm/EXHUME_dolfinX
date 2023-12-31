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
        i = i + 1;
    new_cells = new_cells.reshape(row, col)

    return new_cells, new_points, noncons2cons


def trimHOPoints(original_cells, original_points):
    used_IDS = np.unique(original_cells.flatten()) #index is new ID, value is old
    i = 0 
    for used_ID in used_IDS:
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
block = 0
IDs = exo.cell_data 

for cell in exo.cells:
    node_list = cell.data
    cell_type = cell.type
    cell_IDs = IDs['cell_id'][block]
    if block == 0:
        cells = node_list
        xdmf_IDs = cell_IDs
        materials = np.zeros((1,len(node_list)))
        num_block_cells = len(node_list)
        
    else:
        cells = np.concatenate((cells, node_list))
        xdmf_IDs = np.concatenate((xdmf_IDs, cell_IDs))
        materials = np.concatenate((materials, block*np.ones((1,len(node_list)))),axis = 1)
        num_block_cells += len(node_list)
        
    block = block+1


# make ExOp a mesh function / attribute of each element
points = exo.points

if cell_type == "tetra4":
    #3d, linear, change cell type name 
    cell_type = "tetra"

elif cell_type == "triangle6":
    #2d, quadratic
    # make file with cell node data 
    cells = np.delete(cells, [3, 4, 5], 1)
    cell_type = "triangle"
    cells, points, HO_nodeIDmap = makeIDsConsecutive(cells, points)
    points = trimHOPoints(cells,points)
elif cell_type == "tetra10":
    #3d, quadratic, change cell type name to tetra
    # make file with cell node data 
    cells = np.delete(cells, [4, 5, 6, 7, 8, 9], 1)
    cell_type = "tetra"
    cells, points, HO_nodeIDmap = makeIDsConsecutive(cells, points)
    points = trimHOPoints(cells,points)




print(">>> Creating new mesh ...")

if cell_type == "triangle" or cell_type == "triangle6":
    new_points = np.delete(points, 2, 1)
    points = new_points


ID_map = np.asarray([xdmf_IDs])

mesh = meshio.Mesh(points=points,cells=[(cell_type, cells)],
    cell_data={"ID_map":ID_map})

file =FILE_OUT + "/tri.xdmf"
meshio.write(file,mesh)

mesh = meshio.Mesh(points=points,cells=[(cell_type, cells)],
    cell_data={"material":materials,})
m_file = FILE_OUT + "/tri_materials.xdmf"
meshio.write(m_file,mesh)
