#! /tools/opt/anaconda3_202011/bin/python3.8
import sys
import argparse
import re
import os
import random
import string
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms
import matplotlib.textpath

def get_parser():

    parser = argparse.ArgumentParser('PopCover for Python.')

    parser.add_argument('-f1', dest='p_f1', help='File name with MHC I input')
    parser.add_argument('-f1p', default=None, dest='p_f1p', help='MHC I textarea input')
    parser.add_argument('-f2', dest='p_f2', help='File name with MHC II input')
    parser.add_argument('-f2p', default=None, dest='p_f2p', help='MHC II textarea input')
    parser.add_argument('-fhla', dest='p_hlafile', help='File name with HLA frequencies')
    parser.add_argument('-fhlap', default=None, dest='p_hlafilep', help='HLA frequency textarea input')
    parser.add_argument('-fprot', dest='p_protfile', help='File with protein sequences')
    parser.add_argument('-fprotp', default=None, dest='p_protfilep', help='Protein sequence textarea input')
    parser.add_argument('-useprot', default=False, dest="p_useprot", help="Use protein sequence input")

    parser.add_argument('-ne', dest='p_ne', default="5", help='Number of epitopes to select')
    parser.add_argument('-nmer', dest='p_nmer', default=15, help='Length of peptides to extract from protein sequences')
    parser.add_argument('-nset', dest='p_nset', default="1", help='Number of peptide sets to select')
    parser.add_argument('-b', dest='p_beta', default="0.01", help='Offset for denominator in score function')
    parser.add_argument('-ming', dest='p_mingencov', default="0.0", help='Minimum genomic coverage to include peptide')

    parser.add_argument('-noco1', dest='p_noco1', default=True, help='MHC I binder data without binding cores')
    parser.add_argument('-noco2', dest='p_noco2', default=False, help='MHC II binder data without binding cores')
    parser.add_argument('-selm', dest='p_method', default="popcover", help='Method of peptide selection')
    parser.add_argument('-phf', dest='p_phenofreq', default=False, help='Use phenotype frequencies for calculation')
    parser.add_argument('-smin', dest='p_submin', default=False, help='Substract min. genomic coverage in score function denominator')
    parser.add_argument('-nore', dest='p_nore', default=False, help='Skip dataset reduction')
    parser.add_argument('-tables', dest='p_tables', default=False, help='Make visualization tables')

    parser.add_argument('-req1', dest='p_req1', default=False, help='Require nested class I binder in selected peptides')
    parser.add_argument('-req2', dest='p_req2', default=False, help='Require nested class II binder in selected peptides')

    return parser

def get_args():
    """Get command line arguments"""
    parser = get_parser()
    args = parser.parse_args()

    return args, parser


def get_hlaref(hlafile):
    """Parse hla frequency reference to dictionary"""
    hlaref = dict()

    with open(hlafile) as infile:

        for line in infile:

            if not line.strip():
                continue

            items = line.rstrip().split()

            if len(items) != 2:
                print("Wrong line format in HLA reference file: ", line, '')
                continue

            hla, freq = items[0], float(items[1])
            phenofreq = 2*freq - freq**2

            hlaref[hla] = {'freq':freq, 'phenofreq':phenofreq}

            # Find the locus
            locreg = re.search(r'(HLA-)?[ABC]|(HLA-)?D[A-Z]+[0-9]', hla)
            if locreg is not None:
                locus = locreg.group(0)
                hlaref[hla]['locus'] = locus

    return hlaref


def parse_input(filename, hlaref, no_cores=False, mhc2=False):
    """ Parse input file.
    Line format:    Peptide   HLA   [genotype]   [binding core]
    If no binding core column is given, the peptide itself is treated as the binding core
    """

    def assign_vars(items):
        """ Correct parsing of the lines according to input format"""
        if len(items) == 2:
            # Treat the peptide itself as a binding core
            pep, hla, gen, core = items[0], items[1], "undef", items[0]

        elif len(items) == 3:
            if no_cores:
                # Treat the peptide itself as a binding core
                pep, hla, gen, core = items[0], items[1], items[2], items[0]
            else:
                # No genotype given
                pep, hla, core, gen = items[0], items[1], items[2], "undef"

        elif len(items) == 4:
             pep, hla, gen, core = items

        else:
            print("Error, wrong input format")
            sys.exit(1)

        return pep, hla, gen, core

    hlalist, genlist, epidict = [], [], {}
    lines, skipped_hla = [], []
    s = 0

    # Parse input file
    with open(filename) as infile:

        for line in infile:
            items = line.strip().split()

            if not items:
                continue

            pep, hla, gen, core = assign_vars(items)

            if hla not in hlaref:
                s += 1
                if hla not in skipped_hla:
                    skipped_hla.append(hla)
                continue

            if hla not in hlalist:
                hlalist.append(hla)
            if gen not in genlist:
                genlist.append(gen)

            # Building the epitope data structure
            if pep not in epidict:
                epidict[pep] = {'combos':{(hla, gen)}, 'hlaset':{hla}, 'genset':{gen}, 'covset':{(core, hla, gen)}, 'nested':[0,0]}

                if not mhc2:
                    epidict[pep]['nested'][0] = 1
                else:
                    epidict[pep]['nested'][1] = 1

            else:
                epidict[pep]['hlaset'].add(hla)
                epidict[pep]['genset'].add(gen)
                epidict[pep]['combos'].add((hla, gen))

                # Add combination to covset
                epidict[pep]['covset'].add((core, hla, gen))

    if skipped_hla:
        print("Skipped %d lines due to these HLAs not in allele frequency:<br />" % s)
        print('\t'.join(skipped_hla))
        print("<br />")

    for hla in hlalist:
        hlaref[hla]['class'] = 2 if mhc2 else 1

    return epidict, hlalist, genlist, hlaref


def reduce_dataset(epidict, peplist):
    """Reduce dataset size using Hobohm 1 algorithm
    Requires a list of peptides sorted descending order
    - first by length, then by number of allele,genotype combinations"""

    unique_dict = {peplist[0]:epidict[peplist[0]]}

    for pep in peplist[1:]:
        is_unique = True
        substring = False

        for uniq in unique_dict:

            # Different lengths: check for substring
            if len(pep) < len(uniq):
                if pep in uniq:
                    is_unique = False
                    substring = True

            # Compare the hla, genotype combinations
            if epidict[pep]['covset'].issubset(unique_dict[uniq]['covset']):
                is_unique = False

            if not is_unique:
                # Transfer info from current peptide to the unique peptide
                if substring:
                    for entry in ['hlaset', 'genset', 'combos', 'covset']:
                        unique_dict[uniq][entry] = unique_dict[uniq][entry].union(epidict[pep][entry])

                    for i in (0,1):
                        if epidict[pep]['nested'][i] == 1:
                            unique_dict[uniq]['nested'][i] = 1

                break

        # Add peptide to unique list
        if is_unique:
            unique_dict[pep] = epidict[pep]

    if len(unique_dict) < len(peplist):
        print("Hobohm 1 dataset reduction yielded {} peptides, down from {}<br />".format(len(unique_dict), len(peplist)))
        print("<br />")

    return unique_dict


def calculate_hla_coverage(covered_hlas, hlaref):
    """Calculate per loci and across loci coverages"""

    locidict = dict()

    for hla in covered_hlas:

        if 'locus' in hlaref[hla]:
            locus = hlaref[hla]['locus']

            if locus not in locidict:
                locidict[locus] = [hla]
            else:
                locidict[locus].append(hla)

    # Calculate within loci coverage
    coverages = {"MHC Class I":{}, 'MHC Class II':{}}
    for locus in sorted(locidict):

        mhc_class = hlaref[locidict[locus][0]]['class']

        uncov = 1
        for hla in locidict[locus]:
            uncov *= (1 - hlaref[hla]['phenofreq'])
        cov = 1 - uncov

        mhc = "MHC Class I" if mhc_class == 1 else 'MHC Class II'

        coverages[mhc][locus] = cov

    # Calculate across loci coverage
    across_covs = {}
    for mhc in coverages:
        covs = list(coverages[mhc].values())

        if covs:
            across_cov = covs[0]
            for cov in covs[1:]:
                across_cov += (1 - across_cov)*cov
        else:
            across_cov = 0

        across_covs[mhc] = across_cov

    return coverages, across_covs

def assign_epitope_coverage(epidict, hlaref, genlist, use_pheno, mingencov):
    """For each epitope, give initial score based on sum of hla frequencies"""

    for pep in epidict:
        epidict[pep]['nhita'] = len(epidict[pep]['hlaset'])
        epidict[pep]['nhita_I'] = len([h for h in epidict[pep]['hlaset'] if hlaref[h]['class'] == 1])
        epidict[pep]['nhita_II'] = (epidict[pep]['nhita'] - epidict[pep]['nhita_I'])
        epidict[pep]['nhitg'] = len(epidict[pep]['genset'])

        # coverage per peptide
        covered_hlas = epidict[pep]['hlaset']
        coverages, across_covs = calculate_hla_coverage(covered_hlas, hlaref)

        epidict[pep]['hlacov_I'] = round(across_covs['MHC Class I'], 4)
        epidict[pep]['hlacov_II'] = round(across_covs['MHC Class II'], 4)

        # Initial score
        epidict[pep]['i_score'] = round((epidict[pep]['hlacov_I'] + epidict[pep]['hlacov_II'])*epidict[pep]['nhitg'], 4)

        epidict[pep]['score'] = epidict[pep]['i_score']

    # apply minimum coverage criteria
    if mingencov != 0.0:
        epidict = {key:value for key,value in epidict.items() if value['nhitg']/len(genlist) >= mingencov}
        print("Got %d peptides after applying genomic coverage threshold of %s.<br />" % (len(epidict), str(mingencov)))

    return epidict


def score_peptides(epidict, hlaref, hla_gen_cov, beta, use_pheno, sub, h_indices, g_indices):
    """ Assign score to each peptide for the current selection round and update the epitope dict"""
    for pep in epidict:
        score = 0
        for (hla, gen) in epidict[pep]['combos']:

            f = hlaref[hla]['phenofreq'] if use_pheno else hlaref[hla]['freq']

            h = h_indices[hla]
            g = g_indices[gen]

            score += f / (hla_gen_cov[g][h] + beta - sub)

        epidict[pep]['score'] = round(score, 4)

    return epidict


def select_peptides(e, epidict, hlaref, hlalist, genlist, beta, use_pheno, method, sub, h_indices, g_indices):
    """Iteratively select n peptides by use of the scoring scheme."""

    sel_peps = {}   # dict of selected peptides' dicts
    hla_gen_cov = [[0 for _ in range(len(hlalist))] for _ in range(len(genlist))] # E_ik value matrix

    hla_cov = [0 for _ in range(len(hlalist))] # Counter for each allele
    gen_cov = [0 for _ in range(len(genlist))] # Counter for each genotype

    # Matrices for accumulative coverage count rows
    hla_cov_matrix = []
    gen_cov_matrix = []

    n_combos = len(hlalist)*len(genlist)

    for i in range(e):

        if method == 'random':
            best_pep = random.choice(list(epidict.keys()))

        # Select only based on initally given scores
        elif method == 's_ini':
            best_pep = max(epidict, key=lambda pep:epidict[pep]['i_score'])

        # Normal PopCover method
        else:
            # don't use scoring scheme for first selection
            if i == 0:
                best_pep = max(epidict, key=lambda pep:(epidict[pep]['i_score']))
                #best_pep = max(epidict, key=lambda pep:len(epidict[pep]['hlaset']))
            else:
                epidict = score_peptides(epidict, hlaref, hla_gen_cov, beta, use_pheno, sub, h_indices, g_indices)
                best_pep = max(epidict, key=lambda pep:(epidict[pep]['score'], epidict[pep]['nhita'], epidict[pep]['nhitg']))

        pepdict = epidict[best_pep]

        # Individual coverage
        hvec = [1 if hla in pepdict['hlaset'] else 0 for hla in hlalist]
        gvec = [1 if gen in pepdict['genset'] else 0 for gen in genlist]

        # Accumulative coverage
        hla_cov = [sum(pair) for pair in zip(hla_cov, hvec)]
        gen_cov = [sum(pair) for pair in zip(gen_cov, gvec)]

        hla_cov_matrix.append(hla_cov)
        gen_cov_matrix.append(gen_cov)

        print("Selected %s <br />" % best_pep)
        print("HLA: ", ' '.join([str(n) for n in hla_cov]), '<br />')
        print("Genotype: ", ' '.join([str(n) for n in gen_cov]), '<br />')

        # Update the (hla, gen) combi coverage values
        for (hla, gen) in pepdict['combos']:
            h, g = h_indices[hla], g_indices[gen]
            hla_gen_cov[g][h] += 1

        # Number of covered hla, gen combos
        covered_combos = np.count_nonzero(np.array(hla_gen_cov))

        n_hla = len([n for n in hla_cov if n != 0])
        n_gen = len([n for n in gen_cov if n != 0])

        print("# HLA: {} # Geno: {}".format(n_hla, n_gen))
        print("# HLA+Genotype combo coverage: %i/%i = %.5f <br />" % (covered_combos, n_combos, covered_combos/n_combos))
        print("<br />")

        pepdict['peptide'] = best_pep

        sel_peps[best_pep] = pepdict # Add best peptide's dict to the selections

        del epidict[best_pep]

    return sel_peps, epidict, hla_cov_matrix, gen_cov_matrix


def get_nmers(prot_file, n):
    """ Extract unique n-mer peptides from list of proteins"""
    nmers = set()

    seq = ""

    fasta_format = False

    with open(prot_file) as infile:

        for line in infile:

            if not line.strip():
                continue

            # Header
            if line.startswith('>'):
                fasta_format = True

            if fasta_format:
                if line.startswith('>'):
                    if seq:
                        for i in range(len(seq)):
                            if i < len(seq) - n + 1:
                                nmer = seq[i:i+n]
                                nmers.add(nmer)
                    seq = ""

                else:
                    seq += line.strip().upper()

            # Normal sequence format, one per line
            else:
                seq = line.strip().upper()

                for i in range(len(seq)):
                    if i < len(seq) - n + 1:
                        nmer = seq[i:i+n]
                        nmers.add(nmer)

        # get nmers from the last sequence
        if seq:
            for i in range(len(seq)):
                if i < len(seq) - n + 1:
                    nmer = seq[i:i+n]
                    nmers.add(nmer)

    return nmers


def map_onto_nmers(epidict, mhc_dict):
    """ Map peptide binders onto pre-allocated nmer peptides"""

    for nmer in epidict:

        for pep in mhc_dict:

            if pep in nmer:
                epidict[nmer]['hlaset'] = epidict[nmer]['hlaset'].union(mhc_dict[pep]['hlaset'])
                epidict[nmer]['genset'] = epidict[nmer]['genset'].union(mhc_dict[pep]['genset'])
                epidict[nmer]['combos'] = epidict[nmer]['combos'].union(mhc_dict[pep]['combos'])
                epidict[nmer]['covset'] = epidict[nmer]['covset'].union(mhc_dict[pep]['covset'])

                for i in (0,1):
                    if mhc_dict[pep]['nested'][i] == 1:
                        epidict[nmer]['nested'][i] = 1

    return epidict


def make_html_table(rows, header=[]):
    """Make table to html format"""
    table_lines = ["<table style='text-align: left'>"]

    if header:
        table_lines.append('<tr>')
        for head in header:
            table_lines.append('<th>{}</th>'.format(head))
        table_lines.append('</tr>')

    for row in rows:
        table_lines.append('<tr>')
        for item in row:
            table_lines.append('<td>{}</td>'.format(item))
        table_lines.append('</tr>')

    table_lines.append('</table>')

    return table_lines


def hexcode(rgb):
    """Convert an (r,g,b) tuple to hex color"""
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def main(args, parser):

    mhc1_file = args.p_f1p if args.p_f1p else args.p_f1
    mhc2_file = args.p_f2p if args.p_f2p else args.p_f2
    hla_file = args.p_hlafilep if args.p_hlafilep else args.p_hlafile
    prot_file = args.p_protfilep if args.p_protfilep else args.p_protfile

    # Booleans
    useprot = args.p_useprot
    mhc1_no_cores = args.p_noco1
    mhc2_no_cores = args.p_noco2

    use_pheno = args.p_phenofreq
    nore = args.p_nore
    submin = args.p_submin
    make_tables = args.p_tables

    # popcover, s_ini or random
    method = args.p_method

    req1 = 1 if args.p_req1 and mhc1_file else 0
    req2 = 1 if args.p_req2 and mhc2_file else 0

    nested_classes = (req1, req2)

    try:
        # Floats
        mingencov = float(args.p_mingencov)
        beta = float(args.p_beta)

        # ints
        e = int(args.p_ne)
        nset = int(args.p_nset)

        plen = int(args.p_nmer)

    except ValueError as ex:
        print("Error in numeric options input: ", ex)
        sys.exit(1)

    if hla_file is None:
        print("Please provide input hla frequency file.")
        sys.exit(1)

    if mhc1_file is None and mhc2_file is None:
        print("Please provide input peptide file(s).")
        sys.exit(1)

    # Show parameters
    arg_strings = ["p_ne", "p_nmer", "p_nset", "p_beta", "p_mingencov", "p_useprot", "p_submin", "p_noco1", "p_noco2",
            "p_phenofreq", "p_nore", "p_method"]

    arg_dict = {}
    for item in vars(parser)['_actions']:

        arg = re.search(r"dest='([_\w\d]+)'", str(item)).group(1)
        help_message = re.search(r"help='([\w\W]+)'", str(item)).group(1)

        if arg in arg_strings:
            arg_dict[arg] = [help_message, getattr(args, arg)]

    arg_table = make_html_table([arg_dict[s] for s in arg_strings])

    print("<b>-- Selected options --</b><br />")
    print('\n'.join(arg_table))
    print("<br />")

    # Read in hla frequencies
    hlaref = get_hlaref(hla_file)

    if prot_file and useprot:
        all_nmers = get_nmers(prot_file, plen)

        print("Got %d unique %d-mer peptides from input protein sequences <br/>" % (len(all_nmers), plen))

        epidict = {n:{'hlaset':set(), 'genset':set(), 'combos':set(), 'covset':set(), 'nested':[0,0]} for n in all_nmers}
    else:
        epidict = {}


    if mhc2_file:
        mhc2_dict, hlalist_2, genlist_2, hlaref = parse_input(mhc2_file, hlaref, mhc2_no_cores, mhc2=True)
    else:
        mhc2_dict, hlalist_2, genlist_2 = {}, [], []

    if mhc1_file:
        mhc1_dict, hlalist_1, genlist_1, hlaref = parse_input(mhc1_file, hlaref, mhc1_no_cores)
    else:
        mhc1_dict, hlalist_1, genlist_1 = {}, [], []

    hlalist = sorted(hlalist_1) + sorted(hlalist_2)
    genlist = sorted(list(set(genlist_1 + genlist_2)))

    print("<b>-- MHC binder data --</b><br />")
    print("Got %d alleles <br />" % len(hlalist))
    print("Got %d genotypes <br />" % len(genlist))
    print("<br />")

    # Preallocate hla and genotype indexes for easy lookup of matrix values
    h_indices = {hla:i for i, hla in enumerate(hlalist)}
    g_indices = {gen:i for i, gen in enumerate(genlist)}

    if prot_file and useprot:
        max_len = max([len(p) for p in mhc2_dict] + [len(p) for p in mhc1_dict])

        if plen < max_len:
            print("Error: nmer length smaller than the longest input binder. Defaulting to length of longest binder.<br />")
            plen = max_len

        if mhc2_dict:
            epidict = map_onto_nmers(epidict, mhc2_dict)
        if mhc1_dict:
            epidict = map_onto_nmers(epidict, mhc1_dict)

        # Remove "empty" epitopes
        for pep in [pep for pep in epidict]:
            if epidict[pep] == {'hlaset':set(), 'genset':set(), 'combos':set(), 'covset':set(), 'nested':[0,0]}:
                del epidict[pep]
        print("Got %d unique %d-mer peptides with nested HLA binders <br/>" % (len(epidict), plen))
        print("<br />")

        if not nore:
            # Sorting for hobohm 1
            peplist = sorted([p for p in epidict], key=lambda p: (len(epidict[p]['covset']), epidict[p]['covset'], p), reverse=True)
        else:
            peplist = []

    else:
        epidict = merge_two_dicts(mhc2_dict, mhc1_dict)

        if not nore:
            peplist = sorted([p for p in epidict], key=lambda p: (len(p), len(epidict[p]['covset']), epidict[p]['covset'], p), reverse=True)
        else:
            peplist = []

    if not nore:
        # Reduce with hobohm 1
        epidict = reduce_dataset(epidict, peplist)

    if e > len(epidict):
        e = min([5, len(epidict)])
        print("Error: reduced dataset has less than amount of peptides to select. Defaulting to {}.<br />".format(e))

    epidict = assign_epitope_coverage(epidict, hlaref, genlist, use_pheno, mingencov)

    # subtracting min genomic coverage from popcover score function denominator
    sub = mingencov if submin else 0.0
    if sub == beta:
        sub = 0
        print("Error: genotype coverage threshold equal to beta, will not be subtracted in 1/n denominator.<br />")
    if beta <= 0:
        print("Error: beta has to be bigger than 0. Setting to 0.01 by default.<br />")
        beta = 0.01

    # Apply nested class peptide filter
    if nested_classes != (0,0):

        epidict = {pep:epidict[pep] for pep in epidict if not any(epidict[pep]['nested'][i] == 0 and nested_classes[i] == 1 for i in range(2))}

        class_string = []
        for i, cl in enumerate(["I","II"]):
            if nested_classes[i] == 1:
                class_string.append(cl)

        print("Got {} unique peptides with nested class {} binders <br/>".format(len(epidict), " & ".join(class_string)))
        print("<br />")


    print('HLA: ', ' '.join(hlalist), '<br />')
    print('<br />')
    print('Genotypes: ', ' '.join(genlist), '<br />')

    # Generate random output file name
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    filename += "_" + time.strftime("%Y%m%d")

    out_dir = "/net/sund-nas.win.dtu.dk/storage/services/www/html/services/PopCover-2.0/tmp/"
    #colormap = 'viridis'
    colormap = "nipy_spectral"

    for set_n in range(1, nset + 1):
        print("<br />")
        print("<b>-- Peptide Set %d --</b><br />" % set_n)

        sel_peps, epidict, hla_cov_matrix, gen_cov_matrix = select_peptides(
            e, epidict, hlaref, hlalist, genlist, beta, use_pheno, method, sub, h_indices, g_indices)

        score_names = ["i_score"] if method == 's_ini' else ["i_score", "score"]
        header = ['peptide'] + score_names + ['hlacov_I', 'hlacov_II', 'nhita_I', 'nhita_II', 'nhitg']

        rows = [[str(d[val]) for val in header] for d in sel_peps.values()]
        table_lines = make_html_table(rows, header)

        print('\n'.join(table_lines))
        print("<br />")

        # Take last rows of accumulative count matrices
        hla_cov = hla_cov_matrix[-1]
        gen_cov = gen_cov_matrix[-1]

        covered_hlas = [hla for i, hla in enumerate(hlalist) if hla_cov[i] > 0]

        # Calculating coverage
        coverages, across_covs = calculate_hla_coverage(covered_hlas, hlaref)

        print("<br />")
        coverage_lines = ["<b>-- Locus coverage --</b><br />"]

        for mhc in coverages:

            if coverages[mhc]:
                coverage_lines.append("{}<br />".format(mhc))

                for locus in coverages[mhc]:
                    coverage_lines.append("Locus {} has coverage {} <br />".format(locus, round(coverages[mhc][locus], 4)))

                coverage_lines.append("Coverage across loci in {}: {}<br />".format(mhc, round(across_covs[mhc], 4)))
                coverage_lines.append("<br />")
        print('\n'.join(coverage_lines))

        # Other metrics
        print("<b>-- Number of covering peptides per allele --</b><br />")
        rows = [[hla, cov] for hla, cov in zip(hlalist, hla_cov)]
        table_lines = make_html_table(rows)
        print('\n'.join(table_lines))
        print("<br />")

        print("<b>-- Number of covering peptides per genotype --</b><br />")
        rows = [[gen, cov] for gen, cov in zip(genlist, gen_cov)]
        table_lines = make_html_table(rows)
        print('\n'.join(table_lines))
        print("<br />")

        print("<hr>")

        pep_strings = list(sel_peps.keys())


        output_file_lines = ["HLA: " + ' '.join(hlalist),"Genotype: " + ' '.join(genlist)]

        for i, pep in enumerate(pep_strings):
            output_file_lines.append("{}\t{}\tHLA: {}\tGenotype: {}".format(i+1, pep,
                " ".join([str(x) for x in hla_cov_matrix[i]]), " ".join([str(x) for x in gen_cov_matrix[i]])))

        txt_filename = out_dir + filename + "_{}.txt".format(set_n)

        # list peptides and their covered alleles
        output_file_lines += [""]+["{}\t{}".format(pep, " ".join(sorted(sel_peps[pep]['hlaset']))) for pep in pep_strings]

        with open(txt_filename, 'w+') as outfile:
            for line in output_file_lines:
                outfile.write(line + '\n')

        os.chmod(txt_filename, 0o666)

        web_dir = "https://services.healthtech.dtu.dk/services/PopCover-2.0/tmp/"
        print("<h2>Downloads</h2>")
        print('<a href="{}" download>Download peptide selection (.txt)</a>'.format(web_dir + filename + "_{}.txt".format(set_n)))
        print("<br>")

        if make_tables:
            # Get distinct colors for every count
            rgblist = [getattr(cm, colormap)(x, bytes=True)[:-1] for x in np.linspace(0, 1, e+1)]
            colordict = {i:col for i, col in enumerate([hexcode(rgb) for rgb in rgblist])}

            # Make pretty selection color tables
            for name, mat, cols in [["hla", hla_cov_matrix, hlalist], ["gen", gen_cov_matrix, genlist]]:

                df = pd.DataFrame(mat, columns=cols, index=pep_strings)

                colormat = [[colordict[v] for v in row] for index, row in df.iterrows()]

                # PNG table

                fsize = 7
                # get max text width in header
                t = matplotlib.textpath.TextPath((0,0), max(cols, key=len), size=fsize)
                bb = t.get_extents()
                h = bb.width # Get text length in pixels

                # set fig size
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_frame_on(False)

                tab = plt.table(cellText=df.values, cellColours=colormat, rowLabels=df.index,
                                colLabels=df.columns, cellLoc='center', rowLoc='center')

                for (row, col), cell in tab.get_celld().items():
                    cell.set_linewidth(1)
                    cell.set_edgecolor('w')
                    cell.set_width(0.025)
                    cell._loc = 'center'
                    cell.get_text().set_family('sans-serif')

                    if (row > 0) and (col > -1):
                        cell.get_text().set_color('white')

                    if row == 0:
                        cell.get_text().set_rotation(90)
                        cell.set_height(h/120)
                    if (row == 0) or (col == -1):
                        cell.get_text().set_weight('bold')

                # set font manually
                tab.auto_set_font_size(False)
                tab.set_fontsize(fsize)

                # draw canvas once
                plt.gcf().dpi = 300
                plt.gcf().canvas.draw()

                # get bounding box of table
                #points = tab.get_window_extent(plt.gcf()._cachedRenderer).get_points()
                points = tab.get_window_extent(renderer=fig.canvas.get_renderer()).get_points()

                # add 10 pixel spacing
                points[0,:] -= 10; points[1,:] += 10
                # get new bounding box in inches
                nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)

                plt.savefig(out_dir +'{}_{}_{}.png'.format(name, filename, set_n), dpi=300, bbox_inches=nbbox)


                # Excel version
                writer = pd.ExcelWriter(out_dir + "{}_{}_{}.xlsx".format(name, filename, set_n), engine='xlsxwriter')
                df.to_excel(writer, startrow=1, header=False, sheet_name="Sheet1")

                workbook  = writer.book
                worksheet = writer.sheets['Sheet1']

                # Add a header format.
                header_format = workbook.add_format({
                    'bold': True,
                    'valign': 'bottom',
                    'align': 'center',
                    'border': 1,
                    'rotation':90,
                    'border_color':"white"})

                pep_format = workbook.add_format({
                    'bold':True,
                    'align': 'center',
                    'border': 1,
                    'border_color':"white"})

                # Write the column headers with the defined format.
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)
                # Write the peptide column with defined format
                for i, idx in enumerate(df.index):
                    worksheet.write(i+1, 0, idx, pep_format)
                # Write the table cells with custom colours
                for i, (index, row) in enumerate(df.iterrows()):
                    for j, val in enumerate(row):

                        cell_format = workbook.add_format({
                        'align': 'center',
                        'font_color':'#f9f9f9',
                        'border': 1,
                        'border_color':"white",
                        'bg_color':colordict[val]})

                        worksheet.write(i+1, j+1, val, cell_format)

                # Set cell widths to narrow
                for i in range(1, len(cols) + 1):
                    worksheet.set_column(i, i, 3)
                # set peptide column width
                worksheet.set_column(0, 0, len(max(pep_strings, key=len)) + 1)
                # Close the Pandas Excel writer and output the Excel file.
                writer.save()

            # txt overview

            print("<h3>Colored visualizations (click on the images for larger versions)</h3>")

            out_path = web_dir + '{}_{}_{}.{}'

            hla_im = out_path.format('hla', filename, set_n, 'png')
            gen_im = out_path.format('gen', filename, set_n, 'png')
            hla_xlsx = out_path.format('hla', filename, set_n, 'xlsx')
            gen_xlsx = out_path.format('gen', filename, set_n, 'xlsx')

            print("<h4>Alleles</h4>")
            print("""<a target="_blank" href="{}"> <img src="{}" style="width:500px"> </a><br>""".format(hla_im, hla_im))
            print('<a href="{}" download>Download excel version</a> <br>'.format(hla_xlsx))
            print("<h4>Genotypes</h4>")
            print("""<a target="_blank" href="{}"> <img src="{}" style="width:500px"> </a><br>""".format(gen_im, gen_im))
            print('<a href="{}" download>Download excel version</a> <br>'.format(gen_xlsx))
            print("<br>")


if __name__=='__main__':
    args, parser = get_args()

    main(args, parser)
