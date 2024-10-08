#!/home/tripp/mambaforge/envs/protflow_new/bin/python
# dependency
import numpy as np
import pandas as pd
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
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.metrics.generic_metric_runner import GenericMetric
from protflow.metrics.ligand import LigandContacts
import protflow.tools.rosetta
import protflow.utils.plotting as plots


num_cycles = 1
num_diffs = 100
scaffold_contigs = '10-50/A94-101/4-20/A34-38/4-20/A47-59/10-50'

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

# import input pdb
poses = protflow.poses.Poses(poses=["input/egfr_cetuxi.pdb"], work_dir=".", jobstarter=cpu_jobstarter)

# define diff options
diff_opts = f"diffuser.T=50 'contigmap.contigs=[{scaffold_contigs} B1-162]' inference.ckpt_override_path=/home/tripp/RFdiffusion/models/Complex_beta_ckpt.pt"

# run rfdiffusion
rfdiffusion.run(poses=poses, prefix="rfdiff", num_diffusions=num_diffs, options=diff_opts, fail_on_missing_output_poses=True)

# calculate rog
rog_calculator.run(poses=poses, prefix="rfdiff_rog")
contacts.run(poses=poses, prefix="rfdiff_contacts",)

# filter
#poses.filter_poses_by_value(score_col="rfdiff_rog_data", value=20, operator="<", prefix="rfdiff_rog", plot=True)
#poses.filter_poses_by_value(score_col="rfdiff_contacts_contacts", value=5, operator=">", prefix="rfdiff_contacts", plot=True)


sys.exit()

# run optimization iteratively
for cycle in range(1, num_cycles +1):
    # thread a sequence on binders
    mpnn_opts = f"-fixed_residues {' '.join([f'B{i}' for i in range(1, 163)])}"
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

