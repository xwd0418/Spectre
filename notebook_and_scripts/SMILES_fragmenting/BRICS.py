'''
The RDKit also provides an implementation of the BRICS algorithm. 
BRICS provides a method for fragmenting molecules along synthetically accessible bonds

My plan: 
based on our inference set, create a large set of fragments (by BRICS)
    It is technically a k-way BCE

Our model predicts fragments of a given chemical NMR, and then create mols based on the prediction:
====>

The BRICS module also provides an option to apply the BRICS rules to a set of fragments to create new molecules:
import random
random.seed(127)
fragms = [Chem.MolFromSmiles(x) for x in sorted(allfrags)]
random.seed(0xf00d)
ms = BRICS.BRICSBuild(fragms)
'''

