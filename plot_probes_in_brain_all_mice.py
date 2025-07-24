import numpy as np
import scipy.io
import pandas as pd
import math
from brainrender.actors import Cylinder
from brainrender import Scene
from brainrender import settings
from brainrender.actors import Points
from collections import Counter

from tifffile import imread
reference_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/reference.tiff')
annotations_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/annotation.tiff')
structure_set = pd.read_csv('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/structures.csv')

for i in range(len(structure_set)):
    print(str(structure_set.iloc[i]['acronym']) + " " + str(structure_set.iloc[i]['name']))

class Cylinder2(Cylinder):
    def __init__(self, pos_from, pos_to, root, color='powderblue', alpha=1, radius=350):
        from vedo import shapes
        from brainrender.actor import Actor
        mesh = shapes.Cylinder(pos=[pos_from, pos_to], c=color, r=radius, alpha=alpha)
        Actor.__init__(self, mesh, name="Cylinder", br_class="Cylinder")
# Function to convert stereotaxic coordinates to ABA CCF
# SC is an array with stereotaxic coordinates to be transformed
# Returns an array containing corresponding CCF coordinates in μm
# Conversion is from this post, which explains the opposite transformation: https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858/3
# Warning: this is very approximate
# Warning: the X, Y, Z schematic at the top of the linked post is incorrect, scroll down for correct one.


def StereoToCCF(SC = np.array([1,1,1]), angle = -0.0873):
    # Stretch
    stretch = SC/np.array([1,0.9434,1])
    # Rotate
    rotate = np.array([(stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle)),
                       (stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle)),
                       stretch[2]])
    #Translate
    trans = rotate + np.array([5400, 440, 5700])
    return(trans)


def CCFToStereo(CCF = np.array([1,1,1]), angle = 0.0873):
    #Translate
    trans = CCF - np.array([5400, 440, 5700])
    # Rotate
    rotate = np.array([(trans[0] * math.cos(angle) - trans[1] * math.sin(angle)),
                       (trans[0] * math.sin(angle) + trans[1] * math.cos(angle)),
                       trans[2]])
    # Stretch
    stretch = rotate*np.array([1,0.9434,1])
    return(stretch)


def read_probe_mat(probe_locs_path):
    mat = scipy.io.loadmat(probe_locs_path)
    probe_locs = np.array(mat['probe_locs'])
    return probe_locs
 
def read_borders_table(border_tables_path):
    mat = scipy.io.loadmat(border_tables_path)
    borders_table = pd.DataFrame(mat['borders_table'])
    return borders_table

def adjust_probe_locs(probe_locs):
    adjusted_probe_locs = np.array(
        [[probe_locs[0,0], probe_locs[0,1]], 
        [probe_locs[2,0], probe_locs[2,1]],
        [probe_locs[1,0], probe_locs[1,1]]]
    )*10
    return adjusted_probe_locs
    

def adjust_to_shank_offsets(probe_locs_list_SC, shank_offsets):
    # assumes shank offsets are a df with columns shank and y offset
    # y offset is the displacement of the shank along the directional vector

    corrected_probe_locs_list_SC = np.zeros_like(probe_locs_list_SC)
    corrected_probe_locs_list_CCF = np.zeros_like(probe_locs_list_SC)

    for i, probe_locs_SC in enumerate(probe_locs_list_SC):
        shank_offset = shank_offsets['y_offset'].iloc[i]
        probe_locs_CCF = probe_locs_SC.copy()

        z1, z2 = probe_locs_SC[0]
        y1, y2 = probe_locs_SC[1]
        x1, x2 = probe_locs_SC[2]

        # Calculate the direction vector
        direction_vector = np.array([z2 - z1, 
                                     y2 - y1, 
                                     x2 - x1])

        # Normalize the direction vector
        unit_vector = direction_vector / np.linalg.norm(direction_vector)

        # Scale the unit vector by the distance probe_offset in um
        scaled_vector = unit_vector * shank_offset
        
        # Calculate the new coordinates
        probe_locs_SC[0,1] = z2 + scaled_vector[0]
        probe_locs_SC[1,1] = y2 + scaled_vector[1]
        probe_locs_SC[2,1] = x2 + scaled_vector[2]

        probe_locs_CCF[:,0] = StereoToCCF(probe_locs_SC[:,0])
        probe_locs_CCF[:,1] = StereoToCCF(probe_locs_SC[:,1])

        corrected_probe_locs_list_SC[i, :, :] = probe_locs_SC
        corrected_probe_locs_list_CCF[i, :, :] = probe_locs_CCF
    
    return corrected_probe_locs_list_SC, corrected_probe_locs_list_CCF




def correct_for_left_side(probe_locs_list_CCF, do_correction=True):
    corrected_probe_locs_list_SC = np.zeros_like(probe_locs_list_CCF)
    corrected_probe_locs_list_CCF = np.zeros_like(probe_locs_list_CCF)

    for i, probe_locs_CCF in enumerate(probe_locs_list_CCF):
        probe_locs_SC = probe_locs_CCF.copy()
        probe_locs_SC[:,0] = CCFToStereo(probe_locs_CCF[:,0])
        probe_locs_SC[:,1] = CCFToStereo(probe_locs_CCF[:,1])

        if (probe_locs_SC[:,0][2] > 0) and do_correction:
            probe_locs_SC[:,0][2]*=-1
            probe_locs_SC[:,1][2]*=-1
            probe_locs_CCF[:,0] = StereoToCCF(probe_locs_SC[:,0])
            probe_locs_CCF[:,1] = StereoToCCF(probe_locs_SC[:,1])
        
        corrected_probe_locs_list_SC[i, :, :] = probe_locs_SC
        corrected_probe_locs_list_CCF[i, :, :] = probe_locs_CCF

    return corrected_probe_locs_list_SC, corrected_probe_locs_list_CCF



def main():
    settings.SHADER_STYLE = "glossy"  # other options: metallic, plastic, shiny, glossy, cartoon, default
    settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
    settings.SHOW_AXES = False  # shows/hides the ABA CCF axes from the image
    scene = Scene(root=False, inset=False, atlas_name="allen_mouse_10um")  # makes a scene instance
    root = scene.add_brain_region("root", alpha=0.05, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
    mec = scene.add_brain_region("ENTm", alpha=0.25, color=(106, 202,71), hemisphere="right", silhouette=True)
    #par = scene.add_brain_region("PAR", alpha=0.25, color=(45, 160,23), hemisphere="both", silhouette=True)

    mouse_cluster_annotations_df = pd.DataFrame()
    # load mouse specific probe and cluster spatial locations
    mouse_ids = ["M20", "M21", "M22", "M25", "M26", "M27", "M28", "M29"]

    for Mouse in mouse_ids:
        mouse=int(Mouse.split('M')[1])
        data_paths = [f"/Users/harryclark/Documents/brainrender/probe_data/{Mouse}_probe_locations_{a}.mat" for a in [1,2,3,4]]
        shank_offsets_SC = pd.read_csv('/Users/harryclark/Documents/brainrender/probe_data/shank_offsets.csv')
        shank_offsets_SC = shank_offsets_SC[shank_offsets_SC['mouse'] == mouse]
        probes_locs = [read_probe_mat(data_path) for data_path in data_paths]
        
        # do adjustments
        adjusted_probes_locs_CCF = np.array([adjust_probe_locs(probe_locs) for probe_locs in probes_locs])
        adjusted_probe_locs_SC, adjusted_probe_locs_CCF = correct_for_left_side(adjusted_probes_locs_CCF)
        adjusted_probe_locs_SC, adjusted_probe_locs_CCF = adjust_to_shank_offsets(adjusted_probe_locs_SC, shank_offsets_SC)
        
        # plot probes 
        for i in range(len(adjusted_probe_locs_CCF)):
            actor = Cylinder2(adjusted_probe_locs_CCF[i, :, 0], 
                            adjusted_probe_locs_CCF[i, :, 1], scene.root, color='grey', radius=20)
            scene.add(actor)

    # render
    scene.render(zoom=1.2)
    print("")


if __name__ == '__main__':
    main()
