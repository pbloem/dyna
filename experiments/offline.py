import torch
import torch.nn.functional as F

import dyna
from dyna.util import d

import embed as em

import os, tqdm, fire

from tqdm import trange
"""
The offline baseline. The knowledge embedding model is retrained on the full data each epoch.
"""

def go(name='imdb', epochs=400, lr=3e-4, model='distmult', patience=1, negative_rates=(10, 0, 10),
       batch_size=32, text_batch_size=64, embedding_dim=512, sched=False, losstype='bce', nweight=None):
    """

    :param name:
    :param epochs: Epochs per snapshot.
    :param lr:
    :param model:
    :param nweight: Weight of the negatives in the BCE loss (positives have weight 1)
    :return:
    """

    # Log the hyperparameters
    print(locals())

    # Load the data
    snapshots = dyna.load(name)

    for i, snapshot in enumerate(snapshots):

        print(f'Training on snapshot {i}.')

        i2e, i2r = snapshot['i2e'], snapshot['i2r']

        train = snapshot['train']

        # Collect the corruption candidates for each position in the triple
        subjects   = torch.tensor(list({s for s, _, _ in train}), dtype=torch.long, device=d())
        predicates = torch.tensor(list({p for _, p, _ in train}), dtype=torch.long, device=d())
        objects    = torch.tensor(list({o for _, _, o in train}), dtype=torch.long, device=d())
        ccandidates = (subjects, predicates, objects)

        train = torch.tensor(train, dtype=torch.long, device=d())

        model = em.LinkPredictor(n=snapshot['n'], r=snapshot['r'], embedding=embedding_dim, triples=train, decoder='distmult')
        if torch.cuda.is_available():
            model.cuda()

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=patience, optimizer=opt, mode='max', factor=0.95, threshold=0.0001) \
            if sched else None

        for e in range(epochs):

            model.train(True)

            running, seen = 0.0, 0
            for i in (bar := trange(0, train.size(0), batch_size)):

                opt.zero_grad()
                positives = train[i:i + batch_size].to(d())

                for (s, p, o), labels in em.corruptions(positives, n=len(i2e), r=len(i2r), negative_rates=negative_rates, loss=losstype, ccandidates=ccandidates):

                    scores = model(s, p, o)

                    if losstype == 'bce':
                        # weighting for the negatives
                        cweight = torch.tensor([nweight, 1.0], device=d()) if nweight else None
                        loss = F.binary_cross_entropy_with_logits(scores, labels, weight=cweight, reduction='mean')
                    elif losstype == 'ce':
                        loss = F.cross_entropy(scores, labels, reduction='mean')

                    running += loss.item()
                    seen += 1

                    loss.backward()

                opt.step()
                bar.set_postfix({'loss': running/seen})

if __name__ == '__main__':
    fire.Fire(go)