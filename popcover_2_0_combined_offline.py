import sys
import argparse
import re
import random
import numpy as np

def get_parser():

    parser = argparse.ArgumentParser('PopCover-2.0 for Python.')

    parser.add_argument('-f1', dest='p_f1', help='File name with MHC I input')
    parser.add_argument('-f2', dest='p_f2', help='File name with MHC II input')
    parser.add_argument('-fhla', dest='p_hlafile', help='File name with HLA frequencies')
    parser.add_argument('-fprot', dest='p_protfile', help='File with protein sequences')

    parser.add_argument('-ne', dest='p_ne', default="5", help='Number of epitopes to select')
    parser.add_argument('-nmer', dest='p_nmer', default=15, help='Length of peptides to extract from protein sequences')
    parser.add_argument('-nset', dest='p_nset', default="1", help='Number of peptide sets to select')
    parser.add_argument('-b', dest='p_beta', default="0.01", help='Offset for denominator in score function')
    parser.add_argument('-ming', dest='p_mingencov', default="0.0", help='Minimum genomic coverage to include peptide')

    parser.add_argument('-noco1', dest='p_noco1', action="store_true", help='MHC I binder data without binding cores')
    parser.add_argument('-noco2', dest='p_noco2', action="store_true", help='MHC II binder data without binding cores')
    parser.add_argument('-selm', dest='p_method', default="popcover", help='Method of peptide selection')
    parser.add_argument('-phf', dest='p_phenofreq', action="store_true", help='Use phenotype frequencies for calculation')
    parser.add_argument('-smin', dest='p_submin', action="store_true", help='Substract min. genomic coverage in score function denominator')
    parser.add_argument('-nore', dest='p_nore', action="store_true", help='Skip dataset reduction')

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
                sys.exit(1)

            hla, freq = items[0], float(items[1])
            phenofreq = 2*freq - freq**2

            hlaref[hla] = {'freq':freq, 'phenofreq':phenofreq}

            # Currently DRB3, DRB4, DRB4 are treated as individual loci, which may artificially boost coverage
            locreg = re.search(r'(HLA-)?[ABC]|(HLA-)?D[A-Z]+[0-9]', hla)
            if locreg is not None:
                locus = locreg.group(0)
                hlaref[hla]['locus'] = locus

    print("Got %d alleles from allele frequency list" % len(hlaref))

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
            print("Error, wrong binder input format: ", line)
            sys.exit(1)

        return pep, hla, gen, core

    hlaset, genset, epidict = set(), set(), {}
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

            hlaset.add(hla)
            genset.add(gen)

            # Building the epitope data structure
            if pep not in epidict:
                epidict[pep] = {'combos':{(hla, gen)}, 'hlaset':{hla}, 'genset':{gen}, 'covset':{(core, hla, gen)}}

            else:
                epidict[pep]['hlaset'].add(hla)
                epidict[pep]['genset'].add(gen)
                epidict[pep]['combos'].add((hla, gen))
                epidict[pep]['covset'].add((core, hla, gen))

    if skipped_hla:
        print("Skipped %d lines due to these HLAs not in allele frequency:" % s)
        print('\t'.join(skipped_hla))
        print("")

    for hla in hlaset:
        hlaref[hla]['class'] = 2 if mhc2 else 1

    return epidict, list(hlaset), list(genset), hlaref

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
                break

        # Add peptide to unique list
        if is_unique:
            unique_dict[pep] = epidict[pep]

    if len(unique_dict) < len(peplist):
        print("Hobohm 1 dataset reduction yielded {} peptides, down from {}".format(len(unique_dict), len(peplist)))
        print("")

    return unique_dict


def calculate_hla_coverage(covered_hlas, hlaref):
    """Calculate per loci and across loci coverages"""

    locidict = dict()

    for hla in covered_hlas:

        if 'locus' in hlaref[hla] and 'freq' in hlaref[hla]:
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
        print("Got %d peptides after applying genomic coverage threshold of %s." % (len(epidict), str(mingencov)))

    return epidict


def score_peptides(epidict, hlaref, hla_gen_cov, beta, use_pheno, sub, h_indices, g_indices):
    """ Assign score to each peptide for the current selection round and update the epitope dict"""
    for pep in epidict:

        score = 0
        for (hla, gen) in epidict[pep]['combos']:

            if 'phenofreq' in hlaref[hla]:
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

    n_combos = len(hlalist) * len(genlist)

    for i in range(e):

        if method == 'random':
            best_pep = random.choice(list(epidict.keys()))
        elif method == 's_ini':
            best_pep = max(epidict, key=lambda pep:epidict[pep]['i_score'])

        # Normal PopCover method
        else:
            if i == 0:
                best_pep = max(epidict, key=lambda pep:(epidict[pep]['i_score']))
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

        print("Selected %s " % best_pep)
        print("HLA: ", ' '.join([str(n) for n in hla_cov]), '')
        print("Genotype: ", ' '.join([str(n) for n in gen_cov]), '')

        # Update the (hla, gen) combi coverage values
        for (hla, gen) in pepdict['combos']:
            h, g = h_indices[hla], g_indices[gen]
            hla_gen_cov[g][h] += 1

        # Number of covered hla, gen combos
        covered_combos = np.count_nonzero(np.array(hla_gen_cov))

        n_hla = len([n for n in hla_cov if n != 0])
        n_gen = len([n for n in gen_cov if n != 0])

        print("# HLA: {} # Geno: {}".format(n_hla, n_gen))
        print("# HLA+Genotype combo coverage: %i/%i = %.5f " % (covered_combos, n_combos, covered_combos/n_combos))
        print("")

        pepdict['peptide'] = best_pep

        sel_peps[best_pep] = pepdict # Add best peptide's dict to the selections

        del epidict[best_pep]

    return sel_peps, epidict, hla_cov_matrix, gen_cov_matrix


def get_nmers(prot_file, n):
    """ Extract unique n-mer peptides from list of proteins"""
    nmers = set()
    with open(prot_file) as infile:
        for line in infile:
            # Header
            if line.startswith('>'):
                pass
            else:
                seq = line.strip()

                for i in range(len(seq)):
                    if i < len(seq) - n + 1:
                        nmer = seq[i:i+n]
                        nmers.add(nmer)
    return nmers


def map_onto_nmers(epidict, mhc_dict):
    """ Map peptide binders onto pre-allocated nmer peptides"""
    for nmer in epidict:
        for mhc in mhc_dict:
            if mhc in nmer:
                for entry in ['hlaset', 'genset', 'combos', 'covset']:
                    epidict[nmer][entry] = epidict[nmer][entry].union(mhc_dict[mhc][entry])
    return epidict


def main(args, parser):

    mhc1_file =  args.p_f1
    mhc2_file = args.p_f2
    hla_file = args.p_hlafile
    prot_file = args.p_protfile

    # Booleans
    mhc1_no_cores = args.p_noco1
    mhc2_no_cores = args.p_noco2
    use_pheno = args.p_phenofreq
    nore = args.p_nore
    submin = args.p_submin

    # popcover, s_ini or random
    method = args.p_method

    try:
        mingencov = float(args.p_mingencov)
        beta = float(args.p_beta)
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
    arg_strings = ["p_ne", "p_nmer", "p_nset", "p_beta", "p_mingencov", "p_submin", "p_noco1", "p_noco2",
            "p_phenofreq", "p_nore", "p_method"]

    arg_dict = {}
    for item in vars(parser)['_actions']:

        arg = re.search(r"dest='([_\w\d]+)'", str(item)).group(1)
        help_message = re.search(r"help='([\w\W]+)'", str(item)).group(1)

        if arg in arg_strings:
            arg_dict[arg] = [help_message, str(getattr(args, arg))]

    arg_table = [arg_dict[s] for s in arg_strings]

    print("-- Selected options --")

    format_row = "{:<62} {:<2}"
    for row in arg_table:
        print(format_row.format(*row))
    print("")

    # Read in hla frequencies
    hlaref = get_hlaref(hla_file)
    
    if prot_file:
        all_nmers = get_nmers(prot_file, plen)

        print("Got %d unique %d-mer peptides from input protein sequences " % (len(all_nmers), plen))

        epidict = {n:{'hlaset':set(), 'genset':set(), 'combos':set(), 'covset':set()} for n in all_nmers}
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

    print("-- MHC binder data --")
    print("Got %d alleles " % len(hlalist))
    print("Got %d genotypes " % len(genlist))
    print("")

    # Preallocate hla and genotype indexes for easy lookup of matrix values
    h_indices = {hla:i for i, hla in enumerate(hlalist)}
    g_indices = {gen:i for i, gen in enumerate(genlist)}

    if prot_file:
        max_len = max([len(p) for p in mhc2_dict] + [len(p) for p in mhc1_dict])

        if plen < max_len:
            print("Error: nmer length smaller than the longest input binder. Defaulting to length of longest binder.")
            plen = max_len

        if mhc2_dict:
            epidict = map_onto_nmers(epidict, mhc2_dict)
        if mhc1_dict:
            epidict = map_onto_nmers(epidict, mhc1_dict)

        # Remove "empty" epitopes
        for pep in [p for p in epidict]:
            if epidict[pep] == {'hlaset':set(), 'genset':set(), 'combos':set(), 'covset':set()}:
                del epidict[pep]
        print("Got %d unique %d-mer peptides with nested HLA binders " % (len(epidict), plen))
        print("")

        if not nore:
            # Sorting for hobohm 1
            peplist = sorted([p for p in epidict], key=lambda p: (len(epidict[p]['covset']), epidict[p]['covset'], p), reverse=True)
        else:
            peplist = []

    else:
        epidict = {**mhc2_dict, **mhc1_dict}

        if not nore:
            peplist = sorted([p for p in epidict], key=lambda p: (len(p), len(epidict[p]['covset']), epidict[p]['covset'], p), reverse=True)
        else:
            peplist = []

    if not nore:
        # Reduce with hobohm 1
        epidict = reduce_dataset(epidict, peplist)

    if e > len(epidict):
        e = min([5, len(epidict)])
        print("Error: reduced dataset has less than amount of peptides to select. Defaulting to {}.".format(e))


    epidict = assign_epitope_coverage(epidict, hlaref, genlist, use_pheno, mingencov)

    # subtracting min genomic coverage from popcover score function denominator
    sub = mingencov if submin else 0.0
    if sub == beta:
        sub = 0
        print("Error: genotype coverage threshold equal to beta, will not be subtracted in 1/n denominator.")
    if beta <= 0:
        print("Error: beta has to be bigger than 0. Setting to 0.01 by default.")
        beta = 0.01

    print('HLA: ', ' '.join(hlalist), '')
    print('')
    print('Genotypes: ', ' '.join(genlist), '')

    for set_n in range(1, nset + 1):
        print("")
        print("-- Peptide Set %d --" % set_n)

        sel_peps, epidict, hla_cov_matrix, gen_cov_matrix = select_peptides(
            e, epidict, hlaref, hlalist, genlist, beta, use_pheno, method, sub, h_indices, g_indices)

        score_names = ["i_score"] if method == 's_ini' else ["i_score", "score"]
        header = ['peptide'] + score_names + ['hlacov_I', 'hlacov_II', 'nhita_I', 'nhita_II', 'nhitg']
        rows = [[str(d[val]) for val in header] for d in sel_peps.values()]

        format_row = "{:<16}" + "{:<12}" * (len(rows[0])-1)
        print(format_row.format(*header))
        print('\n'.join(format_row.format(*r) for r in rows))
        print("")

        hla_cov = hla_cov_matrix[-1]
        covered_hlas = [hla for i, hla in enumerate(hlalist) if hla_cov[i] > 0]

        # Calculating coverage    
        coverages, across_covs = calculate_hla_coverage(covered_hlas, hlaref)

        coverage_lines = ["-- Locus coverage --"]

        for mhc in coverages:

            if coverages[mhc]:
                coverage_lines.append("{}".format(mhc))

                for locus in coverages[mhc]:
                    coverage_lines.append("Locus {} has coverage {} ".format(locus, round(coverages[mhc][locus], 8)))

                coverage_lines.append("Coverage across loci in {}: {}".format(mhc, round(across_covs[mhc], 8)))
                coverage_lines.append("")
        print('\n'.join(coverage_lines))


if __name__=='__main__':
    args, parser = get_args()

    main(args, parser)
