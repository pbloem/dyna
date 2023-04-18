import dyna
import fire

"""
Simple script to load and inspect the data
"""

def go(name='imdb'):
    """

    :param name: Name of the data to load (a path to a directory also works)
    :return:
    """

    snapshots = dyna.load(name_or_dir=name)

    for i, snapshot in enumerate(snapshots):
        print(f'Snapshot {i}')
        print(f'     training data has {len(snapshot["train"])} triples')
        print(f'   validation data has {len(snapshot["val"])} triples')
        print(f'         test data has {len(snapshot["test"])} triples')
        print()
        print(f'     some entity labels:', list(snapshot["e2i"].keys())[:5] )
        print(f'   some relation labels:', list(snapshot["r2i"].keys())[:5] )
        print()

if __name__ == '__main__':
    fire.Fire(go)