from .util import here

import os
from os.path import join, isdir
import numpy as np

def rearrange(triples):
    """
    Rearrange from (s, o, p) to (s, p , o)
    """
    return [(s, p, o) for s, o, p in triples]

def load(name_or_dir, verbose=True):
    """
    Load a dynamic KG dataset. This is a knowledge graph that evolved over several snapshot. From one snapshot
    to the next triples may be added or deleted. New entities may appear.

    NB: The entity ids are not necessarily contiguous. To retrieve the largest entity id in a snapshot use
    `max(i2e.keys())', not `len(i2e)`.

    :param name_or_dir:
    :param verbose:
    :return: A list of snapshots, ordered in time. Each is represented by a dicts with the keys:
     * `train`: The training triples: a list of (s, p, o) triples or integer ids.
     * `val: The validation triples.
     * `test`: The test triples.
     * `all`: The union of train, val and test
     * `e2i`, `i2e`: Dictionaries mapping form entity label (e) to integer id (i) and back
     * `r2i`, `i2r`: Dictionaries mapping form relation label (r) to integer id (i) and back
     * `n`: The total number of entities, counting those whose ids aren't used. That is, the largest entity id + 1.
     * `r`: The total number of relations, counting thos whose ids aren't used.
    """

    if name_or_dir == 'imdb':
        dir = here('../data/imdb/')
    else:
        dir = name_or_dir

    i = 0
    snapshots = []

    subpath = join(dir, str(i))
    while isdir(subpath):

        # Load the triples
        train = rearrange(load_tuples(join(subpath, 'train2id_orig.txt'), skiprows=1))
        val   = rearrange(load_tuples(join(subpath, 'valid2id_orig.txt'), skiprows=1))
        test  = rearrange(load_tuples(join(subpath, 'test2id_orig.txt'), skiprows=1))
        all   = rearrange(load_tuples(join(subpath, 'triple2id_orig.txt'), skiprows=1, has_extra_elements=True))

        assert set(train + val + test) == set(all)

        # Load the label to id mappings
        e2i = load_tuples(join(subpath, 'entity2id_orig.txt'), skiprows=1, num_elements=2, to_int=False)
        r2i = load_tuples(join(subpath, 'relation2id_orig.txt'), skiprows=1, num_elements=2, to_int=False)

        # Convert to dicts and reverse
        e2i = {e: int(i) for e, i in e2i}
        r2i = {r: int(i) for r, i in r2i}

        i2e = {i: e for e, i in e2i.items()}
        i2r = {i: r for r, i in r2i.items()}

        snapshots.append({
            'train': train,
            'val': val,
            'test': test,
            'all': all,
            'e2i': e2i, 'i2e': i2e,
            'r2i': r2i, 'i2r': i2r,
            'n': max(i2e.keys())+1,
            'r': max(i2r.keys())+1
        })

        i += 1
        subpath = join(dir, str(i))

    if verbose: print(f'Loaded {len(snapshots)} snapshots from dataset {name_or_dir}.')

    return snapshots

def load_tuples(filepath, num_elements=3, to_int=True, skiprows=0, has_extra_elements=False):
    """
    Loads a series of tuples from a text file. The tuples should be white-space separated, one per line and all of the
    same length.

    :param filepath: The file to load
    :param skiprows: The number of rows to skip before reading the first tuple
    :param num_elements: The number of elements per tuple.
    :param has_extra_elements: Whether the rows are allowed to contain additional elements. These are discarded.
    :param to_int: Whether to cast the individual elements in the tuples to integers. If False, they will be strings.
    :return:
    """

    with open(filepath, 'r') as file:

        lines = file.readlines()[skiprows:]

        if has_extra_elements:
            res = [line.split()[:num_elements] for line in lines]
            assert all([len(line) >= num_elements for line in res]), f'Some lines contain too few elements.'
        else:
            res = [line.split() for line in lines]
            assert all([len(line) == num_elements for line in res]), f'Some lines do not contain exactly {num_elements} elements.'

        if to_int:
            # Cast all elements to integers
            res = [ tuple(int(elem) for elem in line) for line in res]

        return res

