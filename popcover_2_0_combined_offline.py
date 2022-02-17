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
    parser.add_argument('-selm', dest='p_method', default="popcover", help='Method of peptide selection (options: popcover s_ini random)')
    parser.add_argument('-phf', dest='p_phenofreq', action="store_true", help='Use phenotype frequencies for calculation')
    parser.add_argument('-smin', dest='p_submin', action="store_true", help='Substract min. genomic coverage in score function denominator')
    parser.add_argument('-nore', dest='p_nore', action="store_true", help='Skip dataset reduction')
    parser.add_argument('-req1', dest='p_req1', action="store_true", help='Require nested class I binder in selected peptides')
    parser.add_argument('-req2', dest='p_req2', action="store_true", help='Require nested class II binder in selected peptides')
    parser.add_argument('-iedb', dest='p_iedb', action="store_true", help='Use IEDB method for calculating coverage')

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

            if len(items) == 2:
                hla, freq = items

                try:
                    freq = float(freq)
                except ValueError:
                    print(f"Error: invalid allele frequency: {freq}")
                    sys.exit(1)

                phenofreq = 2*freq - freq**2

                hlaref[hla] = {'freq':freq, 'phenofreq':phenofreq, 'hits':0}

                # Find the locus
                locreg = re.search(r'(HLA-)?[ABC]|(HLA-)?D[A-Z]+[0-9]', hla)
                if locreg is not None:
                    locus = locreg.group(0)
                    hlaref[hla]['locus'] = locus

            elif len(items) == 4:
                hla, locus, hla_class, freq = items

                try:
                    freq = float(freq)
                    hla_class = int(hla_class)
                except ValueError:
                    print(f"Error: invalid allele frequency or hla_class: {hla_class} {freq}")
                    sys.exit(1)

                phenofreq = 2*freq - freq**2

                try:
                    hla_class = int(hla_class)
                except ValueError:
                    print("Error: invalid HLA class in reference file, must be either 1 or 2")
                    sys.exit(1)

                hlaref[hla] = {'freq':freq, 'phenofreq':phenofreq, 'hits':0, 'locus':locus, 'class':hla_class}

            else:
                print("Wrong line format in HLA reference file: ", line, '')
                sys.exit(1)

    print(f"Got {len(hlaref)} alleles from allele frequency list")

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
                epidict[pep] = {'combos':{(hla, gen)}, 'hlaset':{hla}, 'genset':{gen}, 'covset':{(core, hla, gen)}, 'nested':[0,0]}

                if not mhc2:
                    epidict[pep]['nested'][0] = 1
                else:
                    epidict[pep]['nested'][1] = 1
            else:
                epidict[pep]['hlaset'].add(hla)
                epidict[pep]['genset'].add(gen)
                epidict[pep]['combos'].add((hla, gen))
                epidict[pep]['covset'].add((core, hla, gen))

    if skipped_hla:
        print(f"Skipped {s} lines due to these HLAs not in allele frequency:")
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

                    for i in (0,1):
                        if epidict[pep]['nested'][i] == 1:
                            unique_dict[uniq]['nested'][i] = 1
                break

        # Add peptide to unique list
        if is_unique:
            unique_dict[pep] = epidict[pep]

    if len(unique_dict) < len(peplist):
        print(f"Hobohm 1 dataset reduction yielded {len(unique_dict)} peptides, down from {len(peplist)}")
        print("")

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
    coverages = {}
    for locus in locidict:
        mhc_class = hlaref[locidict[locus][0]]['class']

        uncov = 1
        for hla in locidict[locus]:
            uncov *= (1 - hlaref[hla]['phenofreq'])
        cov = 1 - uncov

        mhc = f"MHC class {'I'*mhc_class}"

        if mhc not in coverages:
            coverages[mhc] = {}

        coverages[mhc][locus] = cov

    # Calculate across loci coverage
    across_covs = {'MHC class I':0, 'MHC class II':0}
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

        epidict[pep]['hlacov_I'] = round(across_covs['MHC class I'], 4)
        epidict[pep]['hlacov_II'] = round(across_covs['MHC class II'], 4)

        # Initial score
        epidict[pep]['i_score'] = round((epidict[pep]['hlacov_I'] + epidict[pep]['hlacov_II'])*epidict[pep]['nhitg'], 4)

        epidict[pep]['score'] = epidict[pep]['i_score']

    # apply minimum coverage criteria
    if mingencov != 0.0:
        epidict = {key:value for key,value in epidict.items() if value['nhitg']/len(genlist) >= mingencov}
        print(f"Got {len(epidict)} peptides after applying genomic coverage threshold of {mingencov}")

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

        print(f"Selected {best_pep}")
        print("HLA: ", ' '.join([str(n) for n in hla_cov]))
        print("Genotype: ", ' '.join([str(n) for n in gen_cov]))


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
                        for i in range(len(seq) - n + 1):
                            nmer = seq[i:i+n]
                            nmers.add(nmer)
                    seq = ""

                else:
                    seq += line.strip().upper()

            # Normal sequence format, one per line
            else:
                seq = line.strip().upper()

                for i in range(len(seq) - n + 1):
                    nmer = seq[i:i+n]
                    nmers.add(nmer)

        # get nmers from the last sequence
        if seq:
            for i in range(len(seq) - n + 1):
                nmer = seq[i:i+n]
                nmers.add(nmer)

    return nmers

def map_onto_nmers(epidict, mhc_dict):
    """ Map peptide binders onto pre-allocated nmer peptides"""
    for nmer in epidict:
        for pep in mhc_dict:
            if pep in nmer:
                for entry in ['hlaset', 'genset', 'combos', 'covset']:
                    epidict[nmer][entry] = epidict[nmer][entry].union(mhc_dict[pep][entry])

                for i in (0,1):
                    if mhc_dict[pep]['nested'][i] == 1:
                        epidict[nmer]['nested'][i] = 1
    return epidict


def build_locidict(hlaref):

    loci_dict = {}

    for hla in hlaref:
        hla_class, locus = hlaref[hla]['class'], hlaref[hla]['locus']

        if hla_class not in loci_dict:
            loci_dict[hla_class] = {}

        if locus not in loci_dict[hla_class]:
            loci_dict[hla_class][locus] = {hla}
        else:
            loci_dict[hla_class][locus].add(hla)

    return loci_dict

def adjust_frequencies(hlaref, loci_dict):
    """ Make frequencies sum to 1 per loci"""

    for hla_class in loci_dict:
        for locus in loci_dict[hla_class]:
            f_sum = sum(hlaref[hla]['freq'] for hla in loci_dict[hla_class][locus])

            if f_sum > 1:
                for hla in loci_dict[hla_class][locus]:
                    hlaref[hla]['freq'] /= f_sum

            elif f_sum < 1:
                loci_dict[hla_class][locus].add(f'UNKNOWN_{locus}')
                hlaref[f'UNKNOWN_{locus}'] =  {'freq':1-f_sum, 'hits':0}

    return hlaref, loci_dict

def calculate_epitope_hits(epitope_dict, hlaref):
    """Calculate how many epitopes covers each allele"""
    for epitope in epitope_dict:
        for hla in epitope_dict[epitope]:
            hlaref[hla]['hits'] += 1

    return hlaref

def frequency_distribution(loci_dict, hlaref):
    """ For each pair of alleles, calculate their combined epitope hit count and combined frequency
    We then sum the frequencies over the allele pairs with the same hit count, and store them in tabulation_dict
    """
    tabulation_dict = {}

    for hla_class in loci_dict:
        class_dict = {}

        for locus in loci_dict[hla_class]:
            class_dict[locus] = {}

            alleles = list(loci_dict[hla_class][locus])

            for hla_i in alleles:
                i_hits, i_freq = hlaref[hla_i]['hits'], hlaref[hla_i]['freq']

                for hla_j in alleles:
                    j_hits, j_freq = hlaref[hla_j]['hits'], hlaref[hla_j]['freq']

                    if hla_i == hla_j:
                        total_hit = i_hits
                    else:
                        total_hit = i_hits + j_hits

                    # phenotypic frequency
                    phenofreq = i_freq * j_freq

                    if total_hit not in class_dict[locus]:
                        class_dict[locus][total_hit] = phenofreq
                    else:
                        class_dict[locus][total_hit] += phenofreq

        tabulation_dict[hla_class] = class_dict

    return tabulation_dict

def merge_loci(merged_dict, locus_dict):
    """locus_dict is the current loci hit-count/freq map
    to merge onto the current loci(s) """
    merged_locus = {}
    for hit1, freq1 in merged_dict.items():
        for hit2, freq2 in locus_dict.items():
            total_hit = hit1 + hit2
            total_frequency = freq1 * freq2
            if total_hit not in merged_locus:
                merged_locus[total_hit] = total_frequency
            else:
                merged_locus[total_hit] += total_frequency

    return merged_locus


def iedb_coverage(tabulation_dict):

    coverages = {'MHC class I':{}, 'MHC class II':{}}
    across_covs = {'MHC class I':0, 'MHC class II':0}

    # Merge loci for each MHC class
    for mhc_class in tabulation_dict:

        mhc = 'MHC class I' if mhc_class == 1 else 'MHC class II'
        loci = list(tabulation_dict[mhc_class].keys())

        for locus in loci:
            cov = sum(tabulation_dict[mhc_class][locus][h] for h in tabulation_dict[mhc_class][locus] if h > 0)
            coverages[mhc][locus] = cov

        merged_dict = tabulation_dict[mhc_class][loci[0]]

        for locus in loci[1:]:
            locus_dict = tabulation_dict[mhc_class][locus]
            merged_dict = merge_loci(merged_dict, locus_dict)

        coverage = sum(merged_dict[h] for h in merged_dict if h > 0)
        across_covs[mhc] = coverage

    return coverages, across_covs


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

    req1 = 1 if args.p_req1 and mhc1_file else 0
    req2 = 1 if args.p_req2 and mhc2_file else 0

    iedb = args.p_iedb

    nested_classes = (req1, req2)

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
            "p_phenofreq", "p_nore", "p_method", "p_req1", "p_req2", "p_iedb"]

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

        print(f"Got {len(all_nmers)} unique {plen}-mer peptides from input protein sequences")

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

    print("-- MHC binder data --")
    print(f"Got {len(hlalist)} alleles")
    print(f"Got {len(genlist)} genotypes")
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
        for pep in [pep for pep in epidict]:
            if epidict[pep] == {'hlaset':set(), 'genset':set(), 'combos':set(), 'covset':set(), 'nested':[0,0]}:
                del epidict[pep]
        print(f"Got {len(epidict)} unique {plen}-mer peptides with nested HLA binders")
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
        print(f"Error: reduced dataset has less than amount of peptides to select. Defaulting to {e}.")


    epidict = assign_epitope_coverage(epidict, hlaref, genlist, use_pheno, mingencov)

    # subtracting min genomic coverage from popcover score function denominator
    sub = mingencov if submin else 0.0
    if sub == beta:
        sub = 0
        print("Error: genotype coverage threshold equal to beta, will not be subtracted in 1/n denominator.")
    if beta <= 0:
        print("Error: beta has to be bigger than 0. Setting to 0.01 by default.")
        beta = 0.01


    # Apply nested class peptide filter
    if nested_classes != (0,0):
        epidict = {pep:epidict[pep] for pep in epidict if not any(epidict[pep]['nested'][i] == 0 and nested_classes[i] == 1 for i in range(2))}

        class_string = []
        for i, cl in enumerate(["I","II"]):
            if nested_classes[i] == 1:
                class_string.append(cl)

        print("Got {} unique peptides with nested class {} binders".format(len(epidict), " & ".join(class_string)))


    print('HLA: ', ' '.join(hlalist))
    print('')
    print('Genotypes: ', ' '.join(genlist))

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

        # Take last rows of accumulative count matrices
        hla_cov = hla_cov_matrix[-1]
        gen_cov = gen_cov_matrix[-1]

        # Use the iEDB coverage method
        if iedb:
            method = "IEDB"

            loci_dict = build_locidict(hlaref)
            epitope_dict = {pep:sel_peps[pep]['hlaset'] for pep in sel_peps}

            hlaref = calculate_epitope_hits(epitope_dict, hlaref)
            hlaref, loci_dict = adjust_frequencies(hlaref, loci_dict)

            tabulation_dict = frequency_distribution(loci_dict, hlaref)
            coverages, across_covs = iedb_coverage(tabulation_dict)

        # Use the standard PopCover coverage method
        else:
            method = "PopCover"
            covered_hlas = [hla for i, hla in enumerate(hlalist) if hla_cov[i] > 0]
            # Calculating coverage
            coverages, across_covs = calculate_hla_coverage(covered_hlas, hlaref)

        print(f"Population coverage - {method} method")
        print("")
        coverage_lines = []

        for mhc in sorted(coverages):
            if coverages[mhc]:
                coverage_lines.append(mhc)

                for locus in coverages[mhc]:
                    coverage_lines.append(f"Locus {locus} has coverage {round(coverages[mhc][locus], 4)}")

                coverage_lines.append(f"Coverage across loci in {mhc}: {round(across_covs[mhc], 4)}")
        print('\n'.join(coverage_lines))
        print("")

        # Other metrics
        print("-- Number of covering peptides per allele --")
        for hla, count in zip(hlalist, hla_cov):
            print(hla, count )
        print("")

        print("-- Number of covering peptides per genotype--")
        for gen, count in zip(genlist, gen_cov):
            print(gen, count )
        print("")
  

if __name__=='__main__':
    args, parser = get_args()

    main(args, parser)
