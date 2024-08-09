#!/home/tripp/mambaforge/envs/protflow_new/bin/python
# dependency
import numpy as np
import pandas as pd
import os
import sys
import protflow
import protflow.config
from protflow.jobstarters import SbatchArrayJobstarter
import protflow.poses
import protflow.residues
import protflow.tools
import protflow.tools.colabfold
import protflow.tools.esmfold
import protflow.tools.ligandmpnn
import protflow.metrics.rmsd
import protflow.metrics.tmscore
from protflow.metrics.dssp import DSSP
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.metrics.generic_metric_runner import GenericMetric
from protflow.metrics.ligand import LigandContacts
import protflow.tools.rosetta
import protflow.utils.plotting as plots
#from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile, save_structure_to_pdbfile




def main(args):
    hotspot_list = args.hotspot_residues.split(",")

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=100)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = SbatchArrayJobstarter(max_cores=1, gpus=1)

    # set up runners
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    ligandmpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = gpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter = gpu_jobstarter)
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter, fail_on_missing_output_poses=True)
    chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=gpu_jobstarter)

    # set up metrics
    rog_calculator = GenericMetric(module="protflow.utils.metrics", function="calc_rog_of_pdb", jobstarter=small_cpu_jobstarter)
    contacts = LigandContacts(ligand_chain="B", min_dist=0, max_dist=10, atoms=['CA'], jobstarter=small_cpu_jobstarter)
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    rescontacts_calculator = GenericMetric(module="protflow.utils.metrics", function="residue_contacts", jobstarter=small_cpu_jobstarter)
    dssp = DSSP(jobstarter=small_cpu_jobstarter)

    # import input pdb
    poses = protflow.poses.Poses(poses=args.input_dir, glob_suffix="*pdb", work_dir=args.output_dir, jobstarter=cpu_jobstarter)
    
    # define diff options
    diff_opts = f"diffuser.T=50 'contigmap.contigs=[B1-162/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]' inference.ckpt_override_path=/home/tripp/RFdiffusion/models/Complex_beta_ckpt.pt"

    # run rfdiffusion
    rfdiffusion.run(poses=poses, prefix="rfdiff", num_diffusions=args.num_diffs, options=diff_opts, fail_on_missing_output_poses=False)

    # calculate rog, general contacts and hotspot contacts
    rog_calculator.run(poses=poses, prefix="rfdiff_rog")
    contacts.run(poses=poses, prefix="rfdiff_contacts", normalize_by_num_atoms=False)
    dssp.run(poses=poses, prefix="dssp")
    for res in hotspot_list:
        rescontact_opts={"max_distance": 12, "target_chain": "B", "partner_chain": "A", "target_resnum": int(res[1:])+args.binder_length, "target_atom_names": ["CA"], "partner_atom_names": ["CA"]}
        rescontacts_calculator.run(poses=poses, prefix=f"hotspot_{res}_contacts", options=rescontact_opts)

    # calculate overall hotspot contacts
    poses.df["hotspot_contacts"] = sum([poses.df[f"hotspot_{res}_contacts_data"] for res in hotspot_list])

    # filter
    poses.filter_poses_by_value(score_col="rfdiff_rog_data", value=20, operator="<", prefix="rfdiff_rog", plot=True)
    poses.filter_poses_by_value(score_col="rfdiff_contacts_contacts", value=0, operator=">", prefix="rfdiff_contacts", plot=True)
    poses.filter_poses_by_value(score_col="hotspot_contacts", value=20, operator=">", prefix="rfdiff_hotspots_contacts", plot=True)
    for res in hotspot_list:
        poses.filter_poses_by_value(score_col=f"hotspot_{res}_contacts_data", value=1, operator=">", prefix=f"rfdiff_{res}_hotspot_contacts", plot=True)

    # dump output poses
    results_dir = "diffusion_results"
    os.makedirs(results_dir, exist_ok=True)
    poses.save_poses(results_dir)

    if args.skip_optimization:
        #logging.info(f"Skipping optimization. Run concluded, you can probably find the results somewhere around!")
        sys.exit(1)

    # run optimization iteratively
    for cycle in range(1, args.num_opt_cycles +1):
        # thread a sequence on binders
        mpnn_opts = f"-fixed_residues {' '.join([f'B{i}' for i in range(1+args.binder_length, 163+args.binder_length)])}"
        ligandmpnn.run(poses=poses, prefix=f"cycle_{cycle}_seq_thread", nseq=5, model_type="soluble_mpnn", options=mpnn_opts, return_seq_threaded_pdbs_as_pose=True)

        # relax poses
        fr_options = "-parser:protocol fastrelax_interaction.xml -beta"
        rosetta.run(poses=poses, prefix=f"cycle_{cycle}_thread_rlx", nstruct=3, options=fr_options)

        # calculate composite score
        poses.calculate_composite_score(name=f"cycle_{cycle}_threading_comp_score", scoreterms=[f"cycle_{cycle}_thread_rlx_sap_score", f"cycle_{cycle}_thread_rlx_total_score", f"cycle_{cycle}_thread_rlx_interaction_score"], weights=[1,2,3], plot=True)

        # filter to top sequence
        poses.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_threading_comp_score", remove_layers=2)

        # generate sequences for relaxed poses
        ligandmpnn.run(poses=poses, prefix=f"cycle_{cycle}_mpnn", nseq=30, model_type="soluble_mpnn", options=mpnn_opts)

        # write .fasta files (including target) for later use
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_complex_fasta", update_poses=False)

        # remove target chain
        chain_remover.run(poses=poses, prefix=f"cycle_{cycle}_rm_target", chains=["B"])

        # predict
        esmfold.run(poses=poses, prefix=f"cycle_{cycle}_esm")

        # filter for predictions with high confidence
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=80, operator=">", prefix=f"cycle_{cycle}_esm_plddt", plot=True)

        # calculate tm score
        tm_score_calculator.run(poses=poses, prefix=f"cycle_{cycle}_tm", ref_col=f"cycle_{cycle}_thread_rlx_location")

        # filter predictions that don't look like design
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_tm_TM_score_ref", value=0.9, operator=">", prefix=f"cycle_{cycle}_tm_score", plot=True)

        # set .fastas including target as poses
        poses.df["poses"] = poses.df[f"cycle_{cycle}_complex_fasta_fasta_location"]
        poses.parse_descriptions(poses=poses.df["poses"].to_list())

        # predict complexes
        colabfold_opts = "--num-models 3"
        colabfold.run(poses=poses, prefix=f"cycle_{cycle}_af2", options=colabfold_opts, return_top_n_poses=5)

        # filter for predictions with "good" AF2 scores
        #poses.filter_poses_by_value(score_col=f"cycle_{cycle}_af2_plddt", value=90, operator=">", prefix=f"cycle_{cycle}_plddt", plot=True)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    # general optionals
    argparser.add_argument("--skip_optimization", action="store_true", help="Skip the iterative optimization.")
    argparser.add_argument("--num_diffs", type=int, default=100, help="output_directory")
    argparser.add_argument("--num_opt_cycles", type=int, default=1, help="output_directory")
    argparser.add_argument("--hotspot_residues", type=str, default='B18,B39,B41,B108,B131', help="output_directory")
    argparser.add_argument("--binder_length", type=int, default=150, help="output_directory")

    arguments = argparser.parse_args()
    main(arguments)

