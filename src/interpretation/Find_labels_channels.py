from nilearn import datasets
import numpy as np
import nibabel as nib
from scipy.spatial import distance
import pickle as pkl
import nibabel as nib
import numpy as np
from nilearn import datasets


def load_atlas():
    # Load the Harvard-Oxford atlas for example
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')

    # Fetch the atlas map and labels
    atlas_filename = atlas.filename
    atlas_labels = atlas.labels
    return atlas_filename, atlas_labels


# Function to convert voxel coordinates to MNI space
def voxels_to_mni(voxel_coords, affine):
    # Convert voxel coordinates to homogeneous coordinates
    homogeneous_voxels = np.column_stack([voxel_coords, np.ones(voxel_coords.shape[0])])
    # Apply the affine transformation
    mni_coords = homogeneous_voxels.dot(affine.T)[:, :3]
    return mni_coords


def find_mni_regions(atlas_filename, atlas_labels):
    atlas_filename, atlas_labels = load_atlas()
    # Load atlas image
    atlas_img = nib.load(atlas_filename)

    # Get the atlas data and affine matrix
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine  # Affine matrix for voxel-to-MNI conversion

    # Get the voxel coordinates for each region
    voxel_coords = {}
    mni_coords = {}

    for i, label in enumerate(atlas_labels):
        if label == "Background":  # skip the background
            continue
        # Find voxels belonging to the current label
        region_mask = atlas_data == i
        coords = np.column_stack(np.where(region_mask))  # Extract voxel coordinates

        # Store voxel coordinates
        voxel_coords[label] = coords

        # Convert voxel coordinates to MNI space and store them
        mni_coords[label] = voxels_to_mni(coords, affine)
    return mni_coords


def find_label_chennels(mni_region_coor):
    with open('F:/maryam_sh/new_dataset/dataset/move_rest/elec_coor_all_move_rest.pkl', 'rb') as f:
        elec_coor = pkl.load(f)

    ch_names = []
    for patient in range(1):
        print(f'Processing patient {patient}')
        ch_names_p = []
        for i in range(elec_coor[patient].shape[0]):
            dist_all = np.zeros(len(mni_region_coor.keys()))
            for k, region in enumerate(mni_region_coor):
                dist_all[k] = np.mean([distance.euclidean(elec_coor[patient][i, :], mni_region_coor[region][m])
                                       for m in range(mni_region_coor[region].shape[0])])
            ch_names_p.append(list(mni_region_coor.keys())[np.argmin(dist_all)])
        ch_names.append(ch_names_p)
    return ch_names

def preprocess_labels(mni_region_coords):
    list_reg = list(mni_region_coords.keys())
    region_rep = []
    for reg in list_reg:
        if ',' in reg:
            region_rep.append(reg.split(',')[0])
    region_rep = np.unique(region_rep)

    for reg in region_rep:
        reg_base = []
        for i in range(len(list_reg)):
            if list_reg[i].split(',')[0] == reg:
                reg_base.append(mni_region_coords[list_reg[i]])
                del mni_region_coords[list_reg[i]]
        mni_region_coords[reg] = np.concatenate(reg_base)

    del_reg = ['Insular Cortex', 'Angular Gyrus', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
               'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus', 'Precuneous Cortex', 'Cuneal Cortex',
               'Parahippocampal Gyrus', 'Lingual Gyrus',
               'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex']

    for reg in del_reg:
        del mni_region_coords[reg]

    return mni_region_coords
def find_label_chennels2(mni_region_coor):
    with open('F:/maryam_sh/new_dataset/dataset/move_rest/elec_coor_all_move_rest.pkl', 'rb') as f:
        elec_coor = pkl.load(f)

    ch_names = []
    for patient in range(len(elec_coor)):
        print(f'Processing patient {patient}')
        ch_names_p = []
        for i in range(elec_coor[patient].shape[0]):
            dist_all = np.zeros(len(mni_region_coor.keys()))
            for k, region in enumerate(mni_region_coor):
                center_region = np.mean(mni_region_coor[region], axis=0)
                dist_all[k] = distance.euclidean(elec_coor[patient][i, :], center_region)
            ch_names_p.append(list(mni_region_coor.keys())[np.argmin(dist_all)])
        ch_names.append(ch_names_p)
    return ch_names


atlas_filename, atlas_labels = load_atlas()
mni_region_coords = find_mni_regions(atlas_filename, atlas_labels)
mni_region_coords = preprocess_labels(mni_region_coords)
ch_names = find_label_chennels2(mni_region_coords)

np.save('F:/maryam_sh/new_dataset/dataset/move_rest/ch_names.npy', ch_names)
print('end')
