import torch
import torch.nn.functional as F

import dyna
from dyna.util import d

import embed.embed as em
from embed.embed.util import tic, toc

import os, tqdm, fire

from tqdm import trange
import numpy as np

import random

"""
The offline baseline. The knowledge embedding model is retrained on the full data each epoch.
"""


def go(name='imdb', dyna_snapshots=True, epochs=400, lr=0.001, model='distmult', decoder='distmult',patience=9, negative_rates=(10, 0, 10), limit_negatives=False, lred='sum', biases=False,edo=None,rdo=None,
       batch_size=10, test_batch_size=10, val_batch_size=10, val_steps=10, embedding_dim=100, lr_scheduler=True, init_method='uniform',init_parms=(-1,1),
       losstype='ce', optim='adam',momentum=0.0,reciprocal=False, reg_exp=3, reg_eweight=1.55E-10, reg_rweight=3.93E-15, nweight=None,repeats=1,final=False):
    """

    :param name:
    :param epochs: Epochs per snapshot.
    :param lr:
    :param model:
    :param nweight: Weight of the negatives in the BCE loss (positives have weight 1)
    :return:
    """

    global n_repeats
    n_repeats = repeats


    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_mrrs = []

    if dyna_snapshots:
        snapshots=dyna.load(name)
    else:
        raise Exception('The selected dataset has no available snapshots')

    # set of all triples (for filtering)
    for i, snapshot in enumerate(snapshots):
        if name=='imdb':
            i2n, i2r = snapshot['i2e'], snapshot['i2r']
            train,val,test = snapshot['train'],snapshot['val'],snapshot['test']
            train = torch.tensor(train, dtype=torch.long, device=d())
            val = torch.tensor(val, dtype=torch.long, device=d())
            test = torch.tensor(test, dtype=torch.long, device=d())
        else:
            train, val, test, (n2i, i2n), (r2i, i2r) = \
            em.load(name)

        alltriples = set()
        for s, p, o in torch.cat([train, val, test], dim=0):
            s, p, o = s.item(), p.item(), o.item()

            alltriples.add((s, p, o))

        truedicts = em.util.truedicts(alltriples)

        if final:
            train, test = torch.cat([train, val], dim=0), test
        else:
            train, test = train, val

        subjects   = torch.tensor(list({s for s, _, _ in train}), dtype=torch.long, device=d())
        predicates = torch.tensor(list({p for _, p, _ in train}), dtype=torch.long, device=d())
        objects    = torch.tensor(list({o for _, _, o in train}), dtype=torch.long, device=d())
        ccandidates = (subjects, predicates, objects)

        print(snapshot['n'], 'nodes')
        print(snapshot['r'], 'relations')
        print(train.size(0), 'training triples')
        print(test.size(0), 'test triples')
        print(train.size(0) + test.size(0), 'total triples')

        for r in tqdm.trange(n_repeats) if n_repeats > 1 else range(n_repeats):

            """
            Define model
            """
            model = em.LinkPredictor(
                triples=train, n=snapshot['n'], r=snapshot['r'], embedding=embedding_dim, biases=biases,
                edropout = edo, rdropout=rdo, decoder=decoder, reciprocal=reciprocal,
                init_method=init_method, init_parms=init_parms)

            if torch.cuda.is_available():
                model.cuda()

            if optim == 'adam':
                opt = torch.optim.Adam(model.parameters(), lr=lr)
            elif optim == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            elif optim == 'adagrad':
                opt = torch.optim.Adagrad(model.parameters(), lr=lr)
            elif optim == 'sgd':
                opt = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=momentum)
            else:
                raise Exception()

            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=patience, optimizer=opt, mode='max', factor=0.95, threshold=0.0001) \
                if lr_scheduler else None
            #-- defaults taken from libkge

            # nr of negatives sampled
            weight = torch.tensor([nweight, 1.0], device=d()) if nweight else None

            seen = 0
            for e in range(epochs):

                seeni, sumloss = 0, 0.0
                tforward = tbackward = 0
                rforward = rbackward = 0
                tprep = tloss = 0
                tic()

                for fr in trange(0, train.size(0), batch_size):
                    to = min(train.size(0), fr + batch_size)

                    model.train(True)

                    opt.zero_grad()

                    positives = train[fr:to].to(d())

                    for ctarget in [0, 1, 2]: # which part of the triple to corrupt
                        ng = negative_rates[ctarget]

                        if ng > 0:

                            s, p, o, labels = em.corruptions(positives, n=snapshot['n'], r=snapshot['r'], negative_rates=negative_rates, loss=losstype,
                                                     ccandidates=ccandidates, ctarget=ctarget)
                            recip = None if not reciprocal else ('head' if ctarget == 0 else 'tail')
                            # -- We use the tail relations if the target is the relation (usually p-corruption is not used)
                            bs, _ =positives.size()
                            tic()
                            out = model(s, p, o, recip=recip)
                            tforward += toc()

                            assert out.size() == (bs, ng + 1), f'{out.size()=} {(bs, ng + 1)=}'

                            tic()
                            if losstype == 'bce':
                                loss = F.binary_cross_entropy_with_logits(out, labels, weight=weight, reduction=lred)
                            elif losstype == 'ce':
                                loss = F.cross_entropy(out, labels, reduction=lred)

                            assert not torch.isnan(loss), 'Loss has become NaN'

                            sumloss += float(loss.item())
                            seen += bs; seeni += bs
                            tloss += toc()

                            tic()
                            loss.backward()
                            tbackward += toc()
                            # No step yet, we accumulate the gradients over all corruptions.
                            # -- this causes problems with modules like batchnorm, so be careful when porting.

                    tic()
                    regloss = None
                    if reg_eweight is not None:
                        regloss = model.penalty(which='entities', p=reg_exp, rweight=reg_eweight)

                    if reg_rweight is not None:
                        regloss = model.penalty(which='relations', p=reg_exp, rweight=reg_rweight)
                    rforward += toc()

                    tic()
                    if regloss is not None:
                        sumloss += float(regloss.item())
                        regloss.backward()
                    rbackward += toc()

                    opt.step()

                if e == 0:
                    print(f'\n pred: forward {tforward:.4}, backward {tbackward:.4}')
                    print (f'   reg: forward {rforward:.4}, backward {rbackward:.4}')
                    #print (f'           prep {tprep:.4}, loss {tloss:.4}')
                    print (f' total: {toc():.4}')
                    # -- NB: these numbers will not be accurate for GPU runs unless CUDA_LAUNCH_BLOCKING is set to 1

                # Evaluate
                if ((e+1) % val_steps == 0) or e == epochs - 1:

                    with torch.no_grad():

                        model.train(False)

                        if val_batch_size is None:
                            testsub = test
                        else:
                            testsub = test[random.sample(range(test.size(0)), k=val_batch_size)]

                        mrr, hits, ranks = em.util.eval(
                            model=model, valset=testsub, truedicts=truedicts, n=snapshot['n'],
                            batch_size=val_batch_size, verbose=True)

                        print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}')


                        if sched is not None:
                            sched.step(mrr) # reduce lr if mrr stalls

            test_mrrs.append(mrr)

        print('training finished.')

        temrrs = torch.tensor(test_mrrs)
        print(f'mean test MRR    {temrrs.mean():.3} ({temrrs.std():.3})  \t{test_mrrs}')


if __name__ == '__main__':
    fire.Fire(go())
