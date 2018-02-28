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
        percent cutoff for ignoring isotopes

    Returns
    -------
    filtered dictionary
        key=isotope
        value=atomic density
    """
    # check if percent_cutoff value is valid
    if percent_cutoff < 0 or percent_cutoff > 100:
        raise ValueError('Percent has to be between 0 and 100')

    # calculate atomic_density_cutoff
    total_atomic_density = sum(comp_dict.values())
    atomic_density_cutoff = percent_cutoff * total_atomic_density / 100

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
        percent cutoff for ignoring isotopes

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


def csv_render_list_dict(csv_filename, list_dict):
    """renders csv given list of data
       column 1 = entry number, column 2+ = values

    Parameters
    ----------
    csv_filename: str
        path of csv file to be created
    list_dict: dictionary
        dictionary with lists of values to be rendered

    Returns
    -------
    true if successful
    """
    with open(csv_filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write header
        header_list = ['number']
        header_list.extend(list(list_dict.keys()))
        writer.writerow(header_list)

        # check if lengths of all lists are the same
        length_list = []
        key_list = []
        for key, value in list_dict.items():
           length_list.append(len(value))
           key_list.append(key)
        if len(set(length_list)) != 1:
            raise ValueError('Lists have to be the same length') 

        count = 0
        for i in range(0, length_list[0]):
            temp_list = [count + 1]
            for key_index in range(0, len(key_list)):
                temp_list.append(list_dict[key_list[key_index]][i])
            writer.writerow(temp_list)
            count =+ 1

        return True