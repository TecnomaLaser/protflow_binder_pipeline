#!/home/tripp/mambaforge/envs/protflow_new/bin/python
# dependency
import numpy as np
import pandas as pd
import os
import logging
import sys
import protflow
import protflow.config
from protflow.jobstarters import SbatchArrayJobstarter
import protflow.poses
import protflow.residues
import protflow.tools
import protflow.tools.colabfold
from protflow.tools.colabfold import calculate_poses_interaction_pae
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
from protflow.utils.plotting import violinplot_multiple_cols
from protflow.residues import ResidueSelection, residue_selection
#from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile, save_structure_to_pdbfile


def ramp_cutoff(start_value, end_value, cycle, total_cycles) -> float:
    if total_cycles == 1:
        return end_value
    step = (end_value - start_value) / (total_cycles - 1)   
    return start_value + (cycle - 1) * step

def extract_length_from_contig(contig) -> int:
    chains = contig.split(";")
    length = 0
    for chain in chains:
        start, end = chain.split("-")
        length += int(end) - int(start[1:]) +1
    return length


def add_sequence_to_fasta(poses, fasta_location:str, sequence_to_add:str) -> None:
    for fasta in poses.df[fasta_location].to_list():
        with open (fasta,"r") as f:
            contents = f.readlines()
        header = contents[0]
        sequence = contents[1]
        # chains are separated by a colon
        new_content = header + sequence + sequence_to_add #+ ":" + ":".join([f"{i}" for i in sequence.split(":")[1:]])
        with open(fasta, "w") as f:
            f.write(new_content)

def get_resnum_chain(res):
    if int(res[1:]) >= 1000:
        resnum = int(res[1:]) - 1000
        chain = "C"
    else:
        resnum = int(res[1:])
        chain = "B"
    return resnum, chain


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "binder_design.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    hotspot_list = args.hotspot_residues.split(",")
    hotspot_residues_original = residue_selection(args.hotspot_residues, delim=",")
    target_length = extract_length_from_contig(args.target_contig)

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=1000)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = SbatchArrayJobstarter(max_cores=5, gpus=1)

    # set up runners
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    ligandmpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = gpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter = gpu_jobstarter)
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter, fail_on_missing_output_poses=True)
    chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=gpu_jobstarter)

    # set up metrics
    rog_calculator = GenericMetric(module="protflow.utils.metrics", function="calc_rog_of_pdb", options={"chain": "A"}, jobstarter=small_cpu_jobstarter)
    contacts = LigandContacts(ligand_chain="B", min_dist=0, max_dist=10, atoms=['CA'], jobstarter=small_cpu_jobstarter)
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    rescontacts_calculator = GenericMetric(module="protflow.utils.metrics", function="residue_contacts", jobstarter=small_cpu_jobstarter)
    dssp = DSSP(jobstarter=small_cpu_jobstarter)

    # import input pdb
    poses = protflow.poses.Poses(poses=args.input_dir, glob_suffix="*pdb", work_dir=args.output_dir, jobstarter=cpu_jobstarter)
    
    # add hotspot residues to poses df
    poses.df["hotspot_residues_original"] = [hotspot_residues_original for _ in poses.poses_list()]
    poses.df["hotspot_residues_postdiffusion"] = [hotspot_residues_original for _ in poses.poses_list()]

    # define diff options
    target_contig = "/0 ".join(args.target_contig.split(";"))
    diff_opts = f"diffuser.T=50 'contigmap.contigs=[{target_contig}/0 {args.binder_length}-{args.binder_length}]' 'ppi.hotspot_res=[{args.hotspot_residues}]' inference.ckpt_override_path=/home/tripp/RFdiffusion/models/Complex_beta_ckpt.pt"

    # run rfdiffusion
    num_diffs = int(args.num_diffs / 5) if args.num_diffs / 5 >= 1 else 1
    rfdiffusion.run(poses=poses, prefix="rfdiff", num_diffusions=num_diffs, multiplex_poses=5, options=diff_opts, fail_on_missing_output_poses=False, update_motifs=["hotspot_residues_postdiffusion"])
    hotspot_residues_postdiffusion = poses.df["hotspot_residues_postdiffusion"].iloc[0]
    
    # calculate rog, general contacts and hotspot contacts
    rog_calculator.run(poses=poses, prefix="rfdiff_rog")
    contacts.run(poses=poses, prefix="rfdiff_contacts", normalize_by_num_atoms=False)
    dssp.run(poses=poses, prefix="dssp")
    for res in hotspot_residues_postdiffusion.to_list():
        rescontact_opts={"max_distance": 12, "target_chain": "B", "partner_chain": "A", "target_resnum": int(res[1:]), "target_atom_names": ["CA"], "partner_atom_names": ["CA"]}
        rescontacts_calculator.run(poses=poses, prefix=f"hotspot_{res}_contacts", options=rescontact_opts)

    # calculate overall hotspot contacts
    poses.df["hotspot_contacts"] = sum([poses.df[f"hotspot_{res}_contacts_data"] for res in hotspot_residues_postdiffusion.to_list()])

    # make some plots of the hotspot_contacts, RFDiffusion output and the secondary structure content
    cols = ["rfdiff_plddt" , "hotspot_contacts"]
    cols = cols + [f"hotspot_{res}_contacts_data" for res in hotspot_residues_postdiffusion.to_list()]
    cols = cols + [col for col in poses.df.columns if col.startswith("dssp") and col.endswith("content")]
    violinplot_multiple_cols(dataframe = poses.df, cols = cols, y_labels =  cols, out_path = os.path.join(poses.plots_dir, "diffusion_scores.png"))

    # filter
    poses.filter_poses_by_value(score_col="rfdiff_rog_data", value=args.rog_cutoff, operator="<=", prefix="rfdiff_rog", plot=True)
    poses.filter_poses_by_value(score_col="rfdiff_contacts_contacts", value=0, operator=">", prefix="rfdiff_contacts", plot=True)
    poses.filter_poses_by_value(score_col="hotspot_contacts", value=args.hotspot_contacts_cutoff, operator=">=", prefix="rfdiff_hotspots_contacts", plot=True)
    poses.filter_poses_by_value(score_col="dssp_L_content", value = 0.25, operator="<", prefix = "L_content", plot = True)
    for res in hotspot_residues_postdiffusion.to_list():
        poses.filter_poses_by_value(score_col=f"hotspot_{res}_contacts_data", value=args.per_hotspot_contacts_cutoff, operator=">=", prefix=f"rfdiff_{res}_hotspot_contacts", plot=True)

    poses.calculate_composite_score(name="comp_score_before_opt", scoreterms=["rfdiff_rog_data", "hotspot_contacts", "dssp_L_content", "rfdiff_contacts_contacts"], weights=[1, -3, 1, -1], plot=True)

    poses.filter_poses_by_rank(score_col = "comp_score_before_opt", n = args.num_opt_input_poses, prefix = "comp_score", plot = True)

    # dump output poses
    results_dir = os.path.join(poses.work_dir, "diffusion_results")
    os.makedirs(results_dir, exist_ok=True)
    poses.save_poses(results_dir)

    if args.skip_optimization:
        sys.exit(1)

    ########################################################## OPTIMIZATION ##########################################################

    # run optimization iteratively
    for cycle in range(1, args.opt_cycles +1):
        logging.info(f"Starting cycle number {cycle}")

        
        # thread a sequence on binders
        if cycle > 1:
            fixed_residues = ' '.join([f'A{i}' for i in range(args.binder_length + 1, len(args.binder_cterm_stub) + 1)]) + ' ' + ' '.join([f'B{i}' for i in range(1, len(args.target_sequence.split(":")[1]) + 1)]) + ' ' + ' '.join([f'C{i}' for i in range(1, len(args.target_sequence.split(":")[2]) + 1)])
            mpnn_opts = f"--fixed_residues {fixed_residues}"
            logging.info(f"Cycle number {cycle}: the fixed residues for LigandMPNN are: {fixed_residues}")
        else:
            fixed_residues = ' '.join([f'B{i}' for i in range(1+args.binder_length, target_length + 1 + args.binder_length)])
            mpnn_opts = f"--fixed_residues {fixed_residues}"
            logging.info(f"Cycle number {cycle}: the fixed residues for LigandMPNN are: {fixed_residues}")

        ligandmpnn.run(poses=poses, prefix=f"cycle_{cycle}_seq_thread", nseq=5, model_type="soluble_mpnn", options=mpnn_opts, return_seq_threaded_pdbs_as_pose=True)

        # relax poses
        fr_options = "-parser:protocol /home/student300/projects/protflow_binder_pipeline/hotspot_binder/create_rosetta_xml/relax_binder.xml -beta"
        rosetta.run(poses=poses, prefix=f"cycle_{cycle}_thread_rlx", nstruct=3, options=fr_options, rosetta_application="rosetta_scripts.default.linuxgccrelease")
        logging.info(f"Cycle number {cycle}: relaxing poses complete")

        # calculate composite score
        # replace sap with shape complementarity & interaction area
        poses.calculate_composite_score(name=f"cycle_{cycle}_threading_comp_score", scoreterms=[f"cycle_{cycle}_thread_rlx_sap_score",  f"cycle_{cycle}_thread_rlx_total_score", 
        f"cycle_{cycle}_thread_rlx_intE_interaction_energy", f"cycle_{cycle}_thread_rlx_shape_complementarity"], 
        weights =  [1,2,2,-1], plot=True) 
        


        # filter to top sequence
        poses.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_threading_comp_score", remove_layers=2)

        # generate sequences for relaxed poses
        ligandmpnn.run(poses=poses, prefix=f"cycle_{cycle}_mpnn", nseq=50, model_type="soluble_mpnn", options=mpnn_opts, return_seq_threaded_pdbs_as_pose=True)
        logging.info(f"Cycle number {cycle}: LigandMPNN for relaxed poses complete.")
        #poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_complex_fasta", update_poses=False)
        
        # if there is a C terminal stub to add to the binder fasta it is added here in the first cycle:
        #if cycle == 1:
        #    add_sequence_to_fasta(poses, f"cycle_{cycle}_complex_fasta_fasta_location", args.binder_cterm_stub)


        # remove target chain
        if cycle > 1:
            chain_remover.run(poses=poses, prefix=f"cycle_{cycle}_rm_target", chains=["B","C"])
        else:
            chain_remover.run(poses=poses, prefix=f"cycle_{cycle}_rm_target", chains=["B"])
      
        # write .fasta files without target
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_fasta", update_poses=True)
        # if there is a C terminal stub to add to the binder fasta it is added here in the first cycle:
        if cycle == 1:
            add_sequence_to_fasta(poses, f"cycle_{cycle}_fasta_fasta_location", args.binder_cterm_stub)

        # predict
        esmfold.run(poses=poses, prefix=f"cycle_{cycle}_esm")
        logging.info(f"Cycle number {cycle}: ESM prediction complete")

        # filter for predictions with high confidence
        esm_plddt_cutoff = ramp_cutoff(args.opt_plddt_cutoff_start, args.opt_plddt_cutoff_end, cycle, args.opt_cycles)
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=esm_plddt_cutoff, operator=">", prefix=f"cycle_{cycle}_esm_plddt", plot=True)

        # calculate tm score
        tm_score_calculator.run(poses=poses, prefix=f"cycle_{cycle}_tm", ref_col=f"cycle_{cycle}_thread_rlx_location")

        # filter predictions that don't look like design
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_tm_TM_score_ref", value=0.9, operator=">", prefix=f"cycle_{cycle}_tm_score", plot=True)

        # calculate composite score
        poses.calculate_composite_score(name=f"cycle_{cycle}_esm_composite_score", scoreterms=[f"cycle_{cycle}_tm_TM_score_ref", f"cycle_{cycle}_esm_plddt"], weights=[-1,-2], plot=True)

        # filter to cycle input poses
        poses.filter_poses_by_rank(n=15, score_col=f"cycle_{cycle}_esm_composite_score", remove_layers=3, plot=True, prefix=f"cycle_{cycle}_esm_comp_per_bb")

        # filter for maximum number of input poses for af2 (use less input poses for each cycle as it takes too much time otherwise)
        poses.filter_poses_by_rank(n=500, score_col=f"cycle_{cycle}_esm_composite_score", prefix=f"cycle_{cycle}_esm_comp", plot=True)

        # set .fastas including target as poses
        #poses.df["poses"] = poses.df[f"cycle_{cycle}_complex_fasta_fasta_location"]
        #poses.parse_descriptions(poses=poses.df["poses"].to_list())


        # add the complete target sequence to the fasta file:
        poses.convert_pdb_to_fasta(prefix=f"cycle_{cycle}_complex_fasta", update_poses=True)
        add_sequence_to_fasta(poses, f"cycle_{cycle}_complex_fasta_fasta_location", args.target_sequence)

        # predict complexes with ColabFold
        colabfold_opts = "--num-models 3 --msa-mode single_sequence"
        colabfold.run(poses=poses, prefix=f"cycle_{cycle}_af2", options=colabfold_opts)
        logging.info(f"Cycle number {cycle}: Colabfold complete")

        # filter for predictions with good AF2 plddt
        af2_plddt_cutoff = ramp_cutoff(args.opt_plddt_cutoff_start, args.opt_plddt_cutoff_end, cycle, args.opt_cycles)
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_af2_plddt", value=af2_plddt_cutoff, operator=">", prefix=f"cycle_{cycle}_af2_plddt", plot=True)

        # first calculate the TM score again:
        tm_score_calculator.run(poses=poses, prefix=f"cycle_{cycle}_af2_tm", ref_col=f"cycle_{cycle}_thread_rlx_location")

        # filter for TM score
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_af2_tm_TM_score_ref", value=0.9, operator=">", prefix=f"cycle_{cycle}_af2_tm_score", plot=True)

        ## to confirm that the binder is at the correct target position check the hotspot contacts:
        # calculate general contacts and hotspot contacts
        tmp = []
        #logging.info(f"hotspot_residues_original: {type(hotspot_residues_original.to_list())}, {type(hotspot_residues_original.to_list()[0])}")
        for res in hotspot_residues_original.to_list():
            logging.info(f"{res}")
            resnum, chain = get_resnum_chain(res)
            logging.info(f"{resnum}, {chain}, {type(chain)}")
            tmp.append([chain, resnum])
            rescontact_opts={"max_distance": 12, "target_chain": chain, "partner_chain": "A", "target_resnum": resnum, "target_atom_names": ["CA"], "partner_atom_names": ["CA"]}
            rescontacts_calculator.run(poses=poses, prefix=f"cycle_{cycle}_hotspot_{chain+str(resnum)}_contacts", options=rescontact_opts)

        logging.info(f"Cycle number {cycle}: the hotspot contacts are {tmp}")

        # calculate overall hotspot contacts
        poses.df[f"cycle_{cycle}_hotspot_contacts"] = sum([poses.df[f"cycle_{cycle}_hotspot_{res}_contacts_data"] for res in hotspot_residues_original.to_list()])

        # filter out all poses where the contact between target and binder is not given (defined by at least 20 contacts):
        poses.filter_poses_by_value(score_col=f"cycle_{cycle}_hotspot_contacts", value=20, operator=">", prefix=f"cycle_{cycle}_hotspots_contacts", plot=True)
        logging.info(f"The cycle_{cycle}_hotspot_contacts {poses.df[f"cycle_{cycle}_hotspot_contacts"]}")
        

        # calculate the PAE interaction:
        poses = calculate_poses_interaction_pae(prefix=f"cycle_{cycle}", poses=poses, pae_list_col=f"cycle_{cycle}_af2_pae_list", 
        binder_start = args.binder_start, binder_end = args.binder_end, target_start = args.target_start, target_end = args.target_end)

        poses.save_scores()

        # next, calculate a composite score:
        poses.calculate_composite_score(
            name = f"cycle_{cycle}_opt_composite_score",
            scoreterms = [f"cycle_{cycle}_hotspot_contacts", f"cycle_{cycle}_af2_tm_TM_score_ref", f"cycle_{cycle}_af2_plddt", f"cycle_{cycle}_af2_iptm", f"cycle_{cycle}_pae_interaction", ],
            weights = [-1, -1, -2, -3, 4],
            plot = True
        )

        # by removing the index layers we can define how many poses of the same backbone we want
        layers = 4
        if cycle > 1:
            layers += 1

        #filter the poses:
        poses.filter_poses_by_rank(
            n = 5,
            score_col = f"cycle_{cycle}_opt_composite_score",
            prefix = f"cycle_{cycle}_opt_composite_score",
            plot = True,
            remove_layers = layers   
        )

        poses.df = poses.df.sort_values(f"cycle_{cycle}_opt_composite_score")
        poses.reindex_poses(prefix=f"cycle_{cycle}_reindex", force_reindex= True, remove_layers = layers)

        # for checking the ouput
        poses.save_poses(os.path.join(poses.work_dir, f"cycle_{cycle}_output"))
        poses.save_scores(os.path.join(poses.work_dir, f"cycle_{cycle}_scores.json"))


    # relax all structures & calculate shape complementarity, interaction area & delta scores
    fr_options = "-parser:protocol /home/student300/projects/protflow_binder_pipeline/hotspot_binder/create_rosetta_xml/relax_binder.xml -beta"
    rosetta.run(poses=poses, prefix=f"final_thread_rlx", nstruct=3, options=fr_options, rosetta_application="rosetta_scripts.default.linuxgccrelease")
    logging.info(f"After all {cycle} cycles are completed final relaxing poses is complete")

    # calculate a final composite score:
    poses.calculate_composite_score(name=f"final_threading_comp_score", scoreterms=[f"final_thread_rlx_sap_score",  f"final_thread_rlx_total_score", 
        f"final_thread_rlx_intE_interaction_energy", f"final_thread_rlx_shape_complementarity", "final_thread_rlx_shape_complementarity_int_area"], 
        weights =  [1,2,2,-1, -2], plot=True)

if __name__ == "__main__":
    import argparse
    # mandatory
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="Input directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    argparser.add_argument("--hotspot_residues", type=str, required=True, help="Hotspot residues on the target separated by ',' (e.g. 'B18,B39,B41,B108,B131')")
    argparser.add_argument("--target_contig", type=str, required=True, help="Contig of the target (e.g. 'B1-162'  or for a dimer e.g. 'B1-162;C1-120')")

    # filters
    argparser.add_argument("--per_hotspot_contacts_cutoff", type=int, default=0, help="Minimum number of contacts for each hotspot residue")
    argparser.add_argument("--rog_cutoff", type=float, default=20, help="Cutoff for radius of gyration post-diffusion.")
    argparser.add_argument("--hotspot_contacts_cutoff", type=int, default=20, help="Minimum total number of contacts for all hotspot residues")

    # general optionals
    argparser.add_argument("--skip_optimization", action="store_true", help="Skip the iterative optimization.")
    argparser.add_argument("--num_diffs", type=int, default=100, help="Number of RFdiffusions.")
    argparser.add_argument("--binder_length", type=int, default=80, help="Starting amino acid of the binder.")
    argparser.add_argument("--binder_start", type=int, default=0, help="Starting amino acid of the binder.")
    argparser.add_argument("--binder_end", type=int, default=80, help="Last amino acid of the binder")
    argparser.add_argument("--target_start", type=int, default=81, help="Starting amino acid of the target.")
    argparser.add_argument("--target_end", type=int, default=200, help="Last amino acid of the target")
    argparser.add_argument("--num_opt_input_poses", type=int, default=150, help="The number of input poses optimized")
    argparser.add_argument("--binder_cterm_stub", type=str, default="", help="Add C-terminal residues to sequences pre-ESM and pre-AF2 predictions in 1 AA letter code (e.g. MGHHHH). For the 218 linker for the CD20 binder design it is GSTSGSGKPGSGEGSTKG")
    argparser.add_argument("--args.target_sequence", type=str, default=":MTTPRNSVNGTFPAEPMKGPIAMQSGPKPLFRRMSSLVGPTQSFFMRESKTLGAVQIMNGLFHIALGGLLMIPAGIYAPICVTVWYPLWGGIMYIISGSLLAATEKNSRKCLVKGKMIMNSLSLFAAISGMILSIMDILNIKISHFLKMESLNFIRAHTPYINIYNCEPANPSEKNSPSTQYCYSIQSLFLGILSVMLIFAFFQELVIAGIVENEWKRTCSRPKSNIVLLSAEEKKEQTIEIKEEVVGLTETSSQPKNEEDIEIIPIQEEEEEETETNFPEPPQDQESSPIENDSSP:MTTPRNSVNGTFPAEPMKGPIAMQSGPKPLFRRMSSLVGPTQSFFMRESKTLGAVQIMNGLFHIALGGLLMIPAGIYAPICVTVWYPLWGGIMYIISGSLLAATEKNSRKCLVKGKMIMNSLSLFAAISGMILSIMDILNIKISHFLKMESLNFIRAHTPYINIYNCEPANPSEKNSPSTQYCYSIQSLFLGILSVMLIFAFFQELVIAGIVENEWKRTCSRPKSNIVLLSAEEKKEQTIEIKEEVVGLTETSSQPKNEEDIEIIPIQEEEEEETETNFPEPPQDQESSPIENDSSP")

    # optimization optionals
    argparser.add_argument("--opt_cycles", type=int, default=3, help="The number of optimization cycles performed.")
    argparser.add_argument("--opt_plddt_cutoff_end", type=float, default=85, help="End value for plddt filter after each optimization cycle. Filter will be ramped from start to end during optimization.")
    argparser.add_argument("--opt_plddt_cutoff_start", type=float, default=70, help="Start value for plddt filter after each optimization cycle. Filter will be ramped from start to end during optimization.")
    arguments = argparser.parse_args()
    main(arguments)
