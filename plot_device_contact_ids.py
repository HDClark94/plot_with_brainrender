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


def reconstruct_shank_id(clusters_df, mouse):
    shank_ids = []
    for index, cluster in clusters_df.iterrows():
        x_pos = cluster['unit_location_x']
        if mouse != 21:
            if x_pos <= 150:
                shank_id = 0
            elif (x_pos > 150 and x_pos <= 400):
                shank_id = 1
            elif (x_pos > 400 and x_pos <= 650):
                shank_id = 2
            elif x_pos > 650:
                shank_id = 3
            shank_ids.append(shank_id)
        # set the reverse shank ids for this mouse as it was 
        # implanted the other way round to all the other mice
        elif mouse == 21:
            if x_pos <= 150:
                shank_id = 3
            elif (x_pos > 150 and x_pos <= 400):
                shank_id = 2
            elif (x_pos > 400 and x_pos <= 650):
                shank_id = 1
            elif x_pos > 650:
                shank_id = 0
            shank_ids.append(shank_id)
    clusters_df['shank_id'] = shank_ids
    return clusters_df

def brain_coord_from_xy(x_pos, y_pos, probe_locs_list_SC, shank_id):

    # for now assume we're on shank 1
    direction_vector = probe_locs_list_SC[shank_id,:,0] - probe_locs_list_SC[shank_id,:,1]
    # Calculate the unit vector
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    # Calculate the position
    brain_coord_SC = probe_locs_list_SC[shank_id,:,1] + (y_pos * unit_vector)
    brain_coord_SC[2] += x_pos
    brain_coord_CCF = StereoToCCF(brain_coord_SC)
    z_CCF,y_CCF,x_CCF = np.round(brain_coord_CCF/10).astype(int)

    return brain_coord_SC, brain_coord_CCF


def euclidean_distance(point1, point2):
    distance = np.sqrt((point2[0] - point1[0])**2 +
                       (point2[1] - point1[1])**2 +
                       (point2[2] - point1[2])**2)
    return distance

def get_annotation_colors(cluster_annotations):
    annotation_colors = []
    for i in range(len(cluster_annotations)):
        if 'ENT' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (106, 202,71)
        elif 'VIS' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (255, 255,71)
        elif 'RSP' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 0 , 255)
        elif 'SUB' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 255,255)
        elif 'PAR' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (45, 160,23)
        elif 'PRE' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (20, 130,83)
        elif 'HPF' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 255, 0)
        else:
            #print(f'I couldnt assign a color for {cluster_annotations[i]}, so I made it red')
            color = '#%02x%02x%02x' % (255, 0, 0)
        annotation_colors.append(color)
    return annotation_colors

def main():
    settings.SHADER_STYLE = "cartoon"  # other options: metallic, plastic, shiny, glossy, cartoon, default
    settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
    settings.SHOW_AXES = True  # shows/hides the ABA CCF axes from the image
    scene = Scene(root=False, inset=False, atlas_name="allen_mouse_10um")  # makes a scene instance
    root = scene.add_brain_region("root", alpha=0.05, color="grey", hemisphere="both", silhouette=True)  # this is the brain outline
    mec = scene.add_brain_region("ENTm", alpha=0.25, color=(106, 202,71), hemisphere="both", silhouette=True)
    par = scene.add_brain_region("PAR", alpha=0.25, color=(45, 160,23), hemisphere="both", silhouette=True)

    mouse_cluster_annotations_df = pd.DataFrame()
    # load mouse specific probe and cluster spatial locations
    mouse_ids = ["M20", "M21", "M22", "M25", "M26", "M27", "M28", "M29"]
    for Mouse in mouse_ids:
        mouse=int(Mouse.split('M')[1])
        data_paths = [f"/Users/harryclark/Documents/brainrender/probe_data/{Mouse}_probe_locations_{a}.mat" for a in [1,2,3,4]]
        shank_offsets_SC = pd.read_csv('/Users/harryclark/Documents/brainrender/probe_data/shank_offsets.csv')
        clusters_df = pd.read_csv(f"/Users/harryclark/Documents/brainrender/probe_data/device_contact_id_to_channel_location.csv")
        clusters_df = clusters_df[clusters_df['mouse'] == mouse]
        if 'y' in list(clusters_df):
            clusters_df = clusters_df.rename(columns={'y': 'unit_location_y'})
        if 'x' in list(clusters_df):
            clusters_df = clusters_df.rename(columns={'x': 'unit_location_x'})
        shank_offsets_SC = shank_offsets_SC[shank_offsets_SC['mouse'] == mouse]
        clusters_df = reconstruct_shank_id(clusters_df, mouse)
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

        # get brain coords for point (0,0) on shank 0, ignore shank translations for now...
        brain_coord_SC, brain_coord_CCF = brain_coord_from_xy(0,0, adjusted_probe_locs_SC, shank_id=0)
        print(f"Point (0,0) on the probe maps to {brain_coord_CCF} in the CCF format.")
        scene.add(Points(np.reshape(brain_coord_CCF, (1,3)), radius=50, colors="blue"))

        # plot clusters along the probes and create an annotation
        cluster_coord_SCs_x = []
        cluster_coord_SCs_y = []
        cluster_coord_SCs_z = []
        cluster_coord_CCFs_x = []
        cluster_coord_CCFs_y = []
        cluster_coord_CCFs_z = []
        cluster_annotations = []

        cluster_coord_CCFs = []
        for index, cluster in clusters_df.iterrows():
            shank_id = int(cluster['shank_id'])
            x_pos = cluster['unit_location_x']
            y_pos = cluster['unit_location_y']

            if y_pos<0:
                print('theres some points below y=0, I will set it to 0 for debugging purposes')
                y_pos=0

            cluster_coord_SC, cluster_coord_CCF = brain_coord_from_xy(0, y_pos, adjusted_probe_locs_SC, shank_id=shank_id)

            # use the CCF coordinates, get an index and look up the annotation in the brainrender allen_brain_10um volume
            z_CCF,y_CCF,x_CCF = np.round(cluster_coord_CCF/10).astype(int)
            cluster_annotation_index = annotations_set[z_CCF,y_CCF,x_CCF]
            if len(structure_set[structure_set['id'] == cluster_annotation_index]) ==1:
                cluster_annotation = structure_set[structure_set['id'] == cluster_annotation_index]['acronym'].iloc[0]
                cluster_annotations.append(cluster_annotation)
            else: # assume out of brain?
                cluster_annotations.append('root')

            z_CCF,y_CCF,x_CCF = cluster_coord_CCF # np.round(cluster_coord_CCF/10).astype(int) for reference indices
            z_SC,y_SC,x_SC = cluster_coord_SC
            cluster_coord_CCFs.append(cluster_coord_CCF)

            cluster_coord_SCs_z.append(z_SC)
            cluster_coord_SCs_y.append(y_SC)
            cluster_coord_SCs_x.append(x_SC)
            cluster_coord_CCFs_z.append(z_CCF)
            cluster_coord_CCFs_y.append(y_CCF)
            cluster_coord_CCFs_x.append(x_CCF)

        cluster_coord_SCs_z = np.array(cluster_coord_SCs_z)
        cluster_coord_SCs_y = np.array(cluster_coord_SCs_y)
        cluster_coord_SCs_x = np.array(cluster_coord_SCs_x)
        cluster_coord_CCFs_z = np.array(cluster_coord_CCFs_z)
        cluster_coord_CCFs_y = np.array(cluster_coord_CCFs_y)
        cluster_coord_CCFs_x = np.array(cluster_coord_CCFs_x)
        cluster_annotations = np.array(cluster_annotations)
        cluster_annotation_colors = get_annotation_colors(cluster_annotations)
        
        clusters_df['Mouse'] = Mouse
        clusters_df['mouse'] = mouse
        clusters_df['coord_SCs_z'] = cluster_coord_SCs_z
        clusters_df['coord_SCs_y'] = cluster_coord_SCs_y
        clusters_df['coord_SCs_x'] = cluster_coord_SCs_x
        clusters_df['coord_CCFs_z'] = cluster_coord_CCFs_z
        clusters_df['coord_CCFs_y'] = cluster_coord_CCFs_y
        clusters_df['coord_CCFs_x'] = cluster_coord_CCFs_x
        clusters_df['brain_region'] = cluster_annotations
        mouse_cluster_annotations_df = pd.concat([mouse_cluster_annotations_df, clusters_df], ignore_index=True)

        scene.add(Points(np.reshape(cluster_coord_CCFs, (len(cluster_coord_CCFs),3)), radius=50, colors=cluster_annotation_colors, alpha=0.4))

    # save points and render
    mouse_cluster_annotations_df.to_csv('/Users/harryclark/Documents/brainrender/probe_data/device_contact_id_annotations.csv')

    annotations = np.array(mouse_cluster_annotations_df['brain_region']).tolist()
    # Ensure all elements are strings
    annotations = [str(annotation) for annotation in annotations]
    # Count the frequency of each string
    string_counts = Counter(annotations)
    # Calculate the percentage of each string
    total_annotations = len(annotations)
    string_percentages = {string: (count / total_annotations) * 100 for string, count in string_counts.items()}
    
    print("Counts and Percentage of each string:")
    for string, count in string_counts.items():
        percentage = string_percentages[string]
        print(f"{string}: {count} ({percentage:.2f}%)")

    for substring in ['ENT','VIS','PRE','HPF','SUB','PAR','SIM','arb','PFL']:
        substring_count = sum(1 for annotation in annotations if substring in annotation)
        substring_percentage = (substring_count / total_annotations) * 100
        print(f"\nCount and Percentage of strings containing '{substring}': {substring_count} ({substring_percentage:.2f}%)")
        
    # render
    scene.render(zoom=1.2)
    print("")


if __name__ == '__main__':
    main()
