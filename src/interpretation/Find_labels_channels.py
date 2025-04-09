import os
import numpy as np
import nibabel as nib
import pickle as pkl
from nilearn import datasets
from scipy.spatial import distance


def load_atlas():
    """
    Load the Harvard-Oxford cortical atlas.

    Returns:
        - atlas_filename: Path to the atlas image.
        - atlas_labels: List of region labels.
    """
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    return atlas.filename, atlas.labels


def voxels_to_mni(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to MNI space.

    Args:
        voxel_coords: Voxel coordinates as an (N, 3) array.
        affine: Affine matrix from NIfTI image.

    Returns:
        MNI coordinates as an (N, 3) array.
    """
    homogeneous_voxels = np.column_stack([voxel_coords, np.ones(voxel_coords.shape[0])])
    return homogeneous_voxels.dot(affine.T)[:, :3]


def find_mni_regions(atlas_filename: str, atlas_labels: list) -> dict:
    """
    Extract MNI coordinates for each region in the atlas.

    Args:
        atlas_filename: Path to the atlas NIfTI file.
        atlas_labels: List of region labels.

    Returns:
        Dictionary mapping each label to its MNI coordinates.
    """
    atlas_img = nib.load(atlas_filename)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    mni_coords = {}
    for i, label in enumerate(atlas_labels):
        if label == "Background":
            continue
        region_mask = atlas_data == i
        coords = np.column_stack(np.where(region_mask))
        mni_coords[label] = voxels_to_mni(coords, affine)

    return mni_coords


def preprocess_labels(mni_region_coords: dict) -> dict:
    """
    Merge region variants by base name (e.g., "Frontal Pole, left" → "Frontal Pole").

    Args:
        mni_region_coords: Dictionary of region names to MNI coordinates.

    Returns:
        Cleaned dictionary with unified region names.
    """
    region_names = list(mni_region_coords.keys())
    base_names = np.unique([name.split(',')[0] for name in region_names if ',' in name])

    for base in base_names:
        combined_coords = []
        to_delete = []

        for name in region_names:
            if name.startswith(base):
                combined_coords.append(mni_region_coords[name])
                to_delete.append(name)

        for name in to_delete:
            del mni_region_coords[name]

        mni_region_coords[base] = np.concatenate(combined_coords)

    # Optionally remove less informative or irrelevant regions
    irrelevant_regions = [
        'Insular Cortex', 'Angular Gyrus',
        'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
        'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus',
        'Precuneous Cortex', 'Cuneal Cortex', 'Parahippocampal Gyrus', 'Lingual Gyrus',
        'Planum Polare', "Heschl's Gyrus (includes H1 and H2)",
        'Planum Temporale', 'Supracalcarine Cortex'
    ]

    for reg in irrelevant_regions:
        mni_region_coords.pop(reg, None)

    return mni_region_coords


def find_label_channels(mni_region_coords: dict, elec_coor_file: str, use_region_center: bool = True) -> list:
    """
    Assign each electrode to the closest brain region based on MNI coordinates.

    Args:
        mni_region_coords: Dictionary of MNI coordinates per region.
        elec_coor_file: Path to the pickle file containing electrode coordinates.
        use_region_center: If True, use region center instead of voxel-wise distance.

    Returns:
        List of lists containing the closest region name per electrode per patient.
    """
    with open(elec_coor_file, 'rb') as f:
        elec_coor = pkl.load(f)

    ch_names_all = []

    for patient_idx, patient_coords in enumerate(elec_coor):
        print(f'Processing patient {patient_idx}')
        ch_names_patient = []

        for ch_coord in patient_coords:
            distances = []

            for region_name, region_coords in mni_region_coords.items():
                if use_region_center:
                    region_center = np.mean(region_coords, axis=0)
                    dist = distance.euclidean(ch_coord, region_center)
                else:
                    dist = np.mean([
                        distance.euclidean(ch_coord, voxel_coord)
                        for voxel_coord in region_coords
                    ])
                distances.append(dist)

            closest_region = list(mni_region_coords.keys())[np.argmin(distances)]
            ch_names_patient.append(closest_region)

        ch_names_all.append(ch_names_patient)

    return ch_names_all


if __name__ == '__main__':
    # File paths
    elec_file_path = 'F:/maryam_sh/new_dataset/dataset/move_rest/elec_coor_all_move_rest.pkl'
    save_path = 'F:/maryam_sh/new_dataset/dataset/move_rest/ch_names.npy'

    # Load and process atlas
    atlas_filename, atlas_labels = load_atlas()
    mni_region_coords = find_mni_regions(atlas_filename, atlas_labels)
    mni_region_coords = preprocess_labels(mni_region_coords)

    # Assign regions to electrodes
    ch_names = find_label_channels(mni_region_coords, elec_file_path, use_region_center=True)

    # Save results
    np.save(save_path, ch_names)
    print('Channel labeling completed and saved.')
