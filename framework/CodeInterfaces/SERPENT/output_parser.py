import numpy as np
import csv
from pathlib import Path


def parse_line(line):
    """parse composition line by deleting whitespace
       and separating the isotope and atomic density

    Parameters
    ----------
    line: str
        line of isotope and composition

    Returns
    -------
    tuple : (str, float)
        (isotope, atomic density)
    """
    
    # remove whitespace in front
    line = line.lstrip()
    isotope, atom_density = line.split("  ")
    return (isotope, float(atom_density))


def filter_trace(comp_dict, percent_cutoff):
    """filters isotopes with less than percent_cutoff
       for easier calculation

    Parameters
    ----------
    comp_dict: dictionary
        key=isotope
        value=atomic density
    percent_cutoff: float
        percent cutoff for ignoring isotopes (0 - 1)

    Returns
    -------
    filtered dictionary
        key=isotope
        value=atomic density
    """
    # check if percent_cutoff value is valid
    if percent_cutoff < 0 or percent_cutoff > 1:
        raise ValueError('Percent has to be between 0 and 1')

    # calculate atomic_density_cutoff
    total_atomic_density = sum(comp_dict.values())
    atomic_density_cutoff = percent_cutoff * total_atomic_density

    # delete list since cannot change dictionary during iteration
    delete_list = []
    for key, atom_density in comp_dict.items():
        if atom_density < atomic_density_cutoff:
            delete_list.append(key)

    # delete the isotopes with less than percent_cutoff
    for isotope in delete_list:
        del comp_dict[isotope]


    return comp_dict



def bumat_read(bumat_file, percent_cutoff):
    """reads serpent .bumat output file and 
       stores the composition in a dictionary

    Parameters
    ----------
    bumat_file: str
        bumat file path
    percent_cutoff: float
        percent cutoff for ignoring isotopes (0 - 1)

    Returns
    -------
    dict
        dictionary of composition
        (key=isotope(ZZAAA), value=atomic density)
    """
    with open(bumat_file) as f:
        comp_lines = f.readlines()[5:]

    comp_dict = {}
    header = comp_lines[0]
    for i in range(1, len(comp_lines)):
        parsed = parse_line(comp_lines[i])
        # isotope as key, atomic density as value
        comp_dict[parsed[0]] = parsed[1]

    comp_dict = filter_trace(comp_dict, percent_cutoff)
    return comp_dict


def search_keff(res_file):
    """searches and returns the mean keff value in the .res file

    Parameters
    ----------
    res_file: str
        path to .res file

    Returns
    -------
    keff_dict: dict
        keff and std (key = keff or sd, value = list of keff or sd)
    """
    with open(res_file) as f:
        lines = f.readlines()

    keff_list = []
    sd_list = []

    for i in range(0, len(lines)):
        if 'IMP_KEFF' in lines[i]:
            keff_list.append(keff_line_parse(lines[i])[0])
            sd_list.append(keff_line_parse(lines[i])[1])

    keff_dict = {}
    keff_dict['keff'] = keff_list 
    keff_dict['sd'] = sd_list
    return keff_dict


def keff_line_parse(keff_line):
    """parses through the ana_keff line in .res file

    Parameters
    ----------
    keff_line: str
        string from .res file listing IMP_KEFF

    Returns
    -------
    tuple
        (mean IMP_KEFF, std deviation of IMP_KEFF)
    """
    start = keff_line.find('=')
    new_keff_line = keff_line[start:]
    start = new_keff_line.find('[')
    end = new_keff_line.find(']')
    
    # +3 and -1 is to get rid of leading and trailing whitespace
    keff_sd = new_keff_line[start + 3:end - 1]
    (keff, sd) = keff_sd.split(' ')
    return (keff, sd)


def csv_render_dict(csv_filename, dictionary, header):
    """renders csv given the dictionary
       column 1 = key, column 2 = value
    
    Parameters
    ----------
    csv_filename: str
        path of csv file to be created
    dictionary: dict
        dictionary to be rendered into csv file
    header: list
        list of length 2 of header strings

    Returns
    -------
    true if successful.
    """
    with open(csv_filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write header
        writer.writerow(header)
        for key, value in dictionary.items():
            writer.writerow([key, value])
    return True

def read_file_into_list(file):
    """ reads file into list, every line as element

    Parameters
    ----------
    file: str
        name of file

    Returns
    -------
    list of contents in the file
    """
    read = open(file, 'r')
    lines = read.readlines()
    list_from_file = []
    for line in lines:
        list_from_file.append(line.strip())
    read.close()
    return list_from_file

def find_deptime(input_file):
    """ finds the deptime from the inputfile

    Parameters
    ----------
    input_file: str
        input file path

    Returns
    -------
    deptime: string
        depletion time in days
    """
    hit = False
    with open(input_file, 'r') as file:
        for line in file:
            if line.split(' ')[0] == 'dep':
                if line.split(' ')[1] != 'daystep':
                    print('Currently can only take daystep')
                    raise ValueError()
                else:
                    hit = True
                    continue
            if hit:
                deptime = line.split(' ')[0]
                break

    return deptime



def make_csv(csv_filename, in_bumat_dict, out_bumat_dict,
             keff_dict, iso_list, input_file):
    """ renders the  csv as filename with the given
        bumat dict and keff dict

    Parameters
    ----------
    csv_filename: str
        filename of csv output
    in_bumat_dict: dictionary
        key: isotope (ZZAAA)
        value: atomic density
    out_bumat_dict: dictionary
        key: isotope (ZZAAA)
        value: atomic density    
    keff_dict: dictionary
        key: 'keff', 'sd'
        value: keff and sd at EOC
    iso_list: list
        list of isotopes to track
    input_file: str
        path of input file
    """

    # parse through, get keff value
    boc_keff = keff_dict['keff'][0]
    eoc_keff = keff_dict['keff'][1]
    deptime = find_deptime(input_file)
    
    with open(csv_filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # fresh iso_list
        header_list = (['f'+iso for iso in iso_list] +
                      ['boc_keff', 'eoc_keff', 'deptime'] +
                      ['d'+iso for iso in iso_list])
        writer.writerow(header_list)
        # initialize as zero
        fresh_adens_list = [0] * len(iso_list)
        dep_adens_list = [0] * len(iso_list)
        for key in in_bumat_dict:
            if key in iso_list:
                index = iso_list.index(key)
                fresh_adens_list[index] = in_bumat_dict[key]
        for key in out_bumat_dict:
            if key in iso_list:
                index = iso_list.index(key)
                dep_adens_list[index] = out_bumat_dict[key] 

        row = fresh_adens_list + [boc_keff, eoc_keff, deptime] + dep_adens_list
        # add keff value to adens list, like header
        writer.writerow(row)


def main(csv_filename, iso_file, bumat_file, resfile):
    """ Main function that puts everything together to create
        a csv file with all the isotopes in iso_file and keff.

    Parameters
    ----------
    csv_filename: str
        path to output csv file
    iso_file: str
        path to file with isotopes to track
    bumat_file: dictionary
        key: isotope (ZZAAA)
        value: atomic density
    resfile: dictionary
        key: 'keff', 'sd'
        value: keff and sd at EOC

    Returns
    -------
    True if successful
    """
    iso_list = read_file_into_list(iso_file)
    bumat_dict = bumat_read(bumat_dict, 1e-7)
    keff_dict = search_keff(keff_dict)
    make_csv(csv_filename, bumat_dict, keff_dict, iso_list)
    return True
