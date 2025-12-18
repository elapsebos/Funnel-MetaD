# -*- coding: utf-8 -*-
import mdtraj as md
import numpy as np
import pickle
import pandas as pd
from Bio import Align
from deeptime.decomposition import TICA
import sys
import glob

# ================= 1. Global Configuration =================

# Topology file
TOP_FILE = 'combox.pdb'

# List of trajectory files to analyze
# Modify this list based on your actual file names
TRAJ_FILES = [
    'rerun1.xtc', 'rerun2.xtc', 'rerun3.xtc',
    'basin1.xtc', 'basin2.xtc', 'basin3.xtc', 'basin4.xtc',
    'basin5.xtc', 'basin6.xtc', 'basin7.xtc', 'basin8.xtc'
]

# Reference Sequence (Bcl-2 Canonical) used for alignment
REF_SEQ = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMR"

# List of Core Pocket Residues (Canonical IDs)
POCKET_CORE_IDS = [
    118, 119, 129,                        # P1
    111, 112, 115, 133, 136, 137, 149, 153, # P2
    104, 108,                             # P3
    98, 100, 103, 144, 145, 148, 202,     # P4
    146                                   # Arg_c
]

# Safety threshold: Ignore residues with ID < MIN_VALID_ID
MIN_VALID_ID = 90 

# TICA Hyperparameters
TICA_LAG = 100
TICA_DIM = 2

# ================= 2. Helper Functions =================

def get_residue_mapping(pdb_file, ref_seq_str):
    """Establishes mapping: {Simulation_Index : Canonical_ID}"""
    print("--- Running Sequence Alignment (Mapping) ---")
    traj = md.load(pdb_file)
    top = traj.topology
    
    three_to_one = {
        'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
        'HIE':'H', 'HID':'H', 'HIP':'H', 'CYX':'C'
    }
    
    sim_seq_str = ""
    sim_indices = []
    for residue in top.residues:
        if residue.is_protein:
            code3 = residue.name.upper()
            sim_seq_str += three_to_one.get(code3, 'X')
            sim_indices.append(residue.index)
            
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(ref_seq_str, sim_seq_str)[0]
    
    aligned_ref, aligned_sim = alignment[0], alignment[1]
    mapping_dict = {}
    ref_counter = 0 # Canonical ID (1-based)
    sim_counter = 0 # Simulation Index (0-based)
    
    for char_ref, char_sim in zip(aligned_ref, aligned_sim):
        if char_ref != '-': ref_counter += 1
        if char_sim != '-':
            if char_ref != '-':
                real_sim_idx = sim_indices[sim_counter]
                mapping_dict[real_sim_idx] = ref_counter
            sim_counter += 1
            
    return mapping_dict

def get_clean_helix_selection(pdb_path, pocket_ids, mapping_dict, min_id_cutoff):
    """Intelligent selection of backbone residues involved in helices."""
    canonical_to_sim = {v: k for k, v in mapping_dict.items()}
    pocket_sim_indices = [canonical_to_sim[pid] for pid in pocket_ids if pid in canonical_to_sim]
    
    if not pocket_sim_indices:
        print("Warning: No pocket residues found in the simulation structure!")
        return []

    traj = md.load(pdb_path)
    dssp = md.compute_dssp(traj, simplified=True)[0]
    
    helices = []
    current_helix = []
    for i, code in enumerate(dssp):
        if code == 'H':
            current_helix.append(i)
        elif current_helix:
            helices.append(current_helix)
            current_helix = []
    if current_helix: helices.append(current_helix)

    selected_indices = set()
    for helix in helices:
        if not set(helix).isdisjoint(pocket_sim_indices):
            selected_indices.update(helix)
            
    final_indices = []
    for idx in selected_indices:
        if idx in mapping_dict:
            c_id = mapping_dict[idx]
            if c_id >= min_id_cutoff:
                final_indices.append(idx)
                
    return sorted(final_indices)

def get_feature_indices(top, backbone_res_indices, sidechain_res_indices):
    """
    Get atom indices for torsions using MDTraj, then filter by residue list.
    """
    feature_meta = [] 
    
    # Create dummy traj for topology access
    dummy_traj = md.Trajectory(xyz=np.zeros((1, top.n_atoms, 3)), topology=top)

    # --- 1. Backbone Phi/Psi ---
    try:
        phi_inds, _ = md.compute_phi(dummy_traj)
        for i, atoms in enumerate(phi_inds):
            res_idx = top.atom(atoms[2]).residue.index
            if res_idx in backbone_res_indices:
                feature_meta.append(('phi', atoms, res_idx))
    except Exception: pass

    try:
        psi_inds, _ = md.compute_psi(dummy_traj)
        for i, atoms in enumerate(psi_inds):
            res_idx = top.atom(atoms[1]).residue.index
            if res_idx in backbone_res_indices:
                feature_meta.append(('psi', atoms, res_idx))
    except Exception: pass

    # --- 2. Sidechain Chi1/Chi2 ---
    try:
        chi1_inds, _ = md.compute_chi1(dummy_traj)
        for i, atoms in enumerate(chi1_inds):
            res_idx = top.atom(atoms[1]).residue.index
            if res_idx in sidechain_res_indices:
                feature_meta.append(('chi1', atoms, res_idx))
    except Exception: pass

    try:
        chi2_inds, _ = md.compute_chi2(dummy_traj)
        for i, atoms in enumerate(chi2_inds):
            res_idx = top.atom(atoms[1]).residue.index
            if res_idx in sidechain_res_indices:
                feature_meta.append(('chi2', atoms, res_idx))
    except Exception: pass

    return feature_meta

def compute_features_for_traj(traj, feature_meta):
    """
    Compute features based on metadata and apply Sin/Cos transformation.
    Uses md.compute_dihedrals for unified processing.
    """
    groups = {'phi': [], 'psi': [], 'chi1': [], 'chi2': []}
    for item in feature_meta:
        groups[item[0]].append(item[1])
    
    data_list = []
    
    # Process order must match generate_clean_labels: Phi -> Psi -> Chi1 -> Chi2
    for key in ['phi', 'psi', 'chi1', 'chi2']:
        indices = np.array(groups[key], dtype=np.int32)
        
        if len(indices) > 0:
            try:
                angles = md.compute_dihedrals(traj, indices)
                # Sin/Cos Transformation
                data_list.append(np.sin(angles))
                data_list.append(np.cos(angles))
            except Exception as e:
                print(f"   [Error] Calculating {key}: {e}")
                return None
            
    if not data_list:
        return None
        
    return np.hstack(data_list)

def generate_clean_labels(feature_meta, mapping_dict, top):
    """Generate labels consistent with the computed data order."""
    labels = []
    keys = ['phi', 'psi', 'chi1', 'chi2']
    
    for key in keys:
        sub_meta = [m for m in feature_meta if m[0] == key]
        if not sub_meta: continue
        
        for func_name in ['sin', 'cos']:
            for item in sub_meta:
                _, _, res_idx = item
                if res_idx in mapping_dict:
                    c_id = mapping_dict[res_idx]
                    res_name = top.residue(res_idx).name.capitalize()
                    label = f"{res_name}{c_id}-{key}-{func_name}"
                else:
                    label = f"Res{res_idx}-{key}-{func_name}"
                labels.append(label)
    return labels

# ================= 3. Main Execution =================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"   Starting TICA Analysis (Deeptime) | Filter ID >= {MIN_VALID_ID}")
    print(f"{'='*60}\n")
    
    # 1. Establish Mapping
    mapping_dict = get_residue_mapping(TOP_FILE, REF_SEQ)
    
    # 2. Define Features
    print("\n[Step 1] Analyzing topology and defining features...")
    top = md.load(TOP_FILE).topology

    backbone_indices = get_clean_helix_selection(TOP_FILE, POCKET_CORE_IDS, mapping_dict, MIN_VALID_ID)
    print(f"   -> Selected backbone residues: {len(backbone_indices)}")

    canonical_to_sim = {v: k for k, v in mapping_dict.items()}
    sidechain_indices = []
    for pid in POCKET_CORE_IDS:
        if pid >= MIN_VALID_ID and pid in canonical_to_sim:
            sidechain_indices.append(canonical_to_sim[pid])
    sidechain_indices = sorted(list(set(sidechain_indices)))
    print(f"   -> Selected sidechain residues: {len(sidechain_indices)}")

    feature_meta = get_feature_indices(top, backbone_indices, sidechain_indices)
    clean_labels = generate_clean_labels(feature_meta, mapping_dict, top)
    print(f"   -> Feature dimensions: {len(clean_labels)} (including Sin/Cos)")

    # 3. Load Data & Compute Features
    print("\n[Step 2] Loading trajectories and computing features (MDTraj)...")
    data_input = []
    
    for trj_file in TRAJ_FILES:
        print(f"   Processing: {trj_file} ...", end='\r')
        try:
            t = md.load(trj_file, top=TOP_FILE)
            feats = compute_features_for_traj(t, feature_meta)
            if feats is not None:
                data_input.append(feats.astype(np.float32))
        except Exception as e:
            print(f"\n   [Error] Failed to read {trj_file}: {e}")
            
    print(f"\n   Data loaded. Number of trajectories: {len(data_input)}")
    if not data_input:
        sys.exit("No valid data found.")

    # 4. Compute TICA
    print("\n[Step 3] Computing TICA (Deeptime) ...")
    estimator = TICA(lagtime=TICA_LAG, dim=TICA_DIM)
    tica_model = estimator.fit(data_input).fetch_model()
    tica_output = [tica_model.transform(traj) for traj in data_input]
    
    # 5. Compute Correlations (Feature Weights)
    print("\n[Step 4] Computing Feature-TIC Correlations ...")
    X_concat = np.concatenate(data_input)
    Y_concat = np.concatenate(tica_output)
    
    # Manual Pearson Correlation Calculation
    X_centered = X_concat - X_concat.mean(axis=0)
    Y_centered = Y_concat - Y_concat.mean(axis=0)
    
    X_std = X_centered.std(axis=0)
    Y_std = Y_centered.std(axis=0)
    X_std[X_std == 0] = 1.0
    Y_std[Y_std == 0] = 1.0
    
    cov = np.dot(X_centered.T, Y_centered) / X_concat.shape[0]
    correlations = cov / np.outer(X_std, Y_std)
    feature_coeffs = correlations

    # 6. Export Data
    print("\n[Step 5] Exporting Data ...")
    
    # --- Export Weights ---
    df_weights = pd.DataFrame({
        'Feature': clean_labels,
        'TIC1_Corr': feature_coeffs[:, 0],
        'TIC2_Corr': feature_coeffs[:, 1],
        'TIC1_Abs': np.abs(feature_coeffs[:, 0])
    })
    df_weights_sorted = df_weights.sort_values(by='TIC1_Abs', ascending=False)
    df_weights_sorted.to_csv('check_weights_deeptime.csv', index=False)
    print(f"   [Saved] Weights: check_weights_deeptime.csv")
    
    # --- Export Projections ---
    all_proj_data = []
    for i, traj_dat in enumerate(tica_output):
        df_tmp = pd.DataFrame(traj_dat, columns=['TIC1', 'TIC2'])
        df_tmp['File'] = TRAJ_FILES[i]
        all_proj_data.append(df_tmp)
    
    df_proj_all = pd.concat(all_proj_data, ignore_index=True)
    df_proj_all.to_csv('check_projection_deeptime.csv', index=False)
    print(f"   [Saved] Projections: check_projection_deeptime.csv")

    # --- Export PKL ---
    pkl_data = {
        'tica_output': tica_output,
        'feature_coeffs': feature_coeffs,
        'labels': clean_labels,
        'traj_files': TRAJ_FILES,
        'model_eigenvalues': tica_model.singular_values,
        'model_timescales': tica_model.timescales
    }
    with open('tica_results_deeptime.pkl', 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"   [Saved] Pickle Data: tica_results_deeptime.pkl")

    print(f"\n{'='*20} Top 5 Features {'='*20}")
    print(df_weights_sorted[['Feature', 'TIC1_Abs']].head(5).to_string(index=False))
    print(f"{'='*60}")
    print("Analysis Completed.")