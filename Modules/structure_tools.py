from typing import List

import copy, sys, os, random

import itertools

from typing import Tuple, List, Dict

import numpy as np

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = False
IPythonConsole.drawOptions.addStereoAnnotation = False
import deepchem as dc

from torch import nn
import torch

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score

from IR_Functions import LoadPickle



####################################### ___ #######################################

# if __name__ == "main":

featurizer=dc.feat.WeaveFeaturizer()

def struct_featurizer(state:Chem.Mol) -> np.ndarray:

    mol = featurizer(state)[0]
    
    x = mol.get_pair_features().astype('bool')
    # print(x.shape)
    y = mol.get_atom_features().astype('bool')
    # print(y.shape)

    x = x.T.reshape((x.shape[1],) + (y.shape[0],)*2)
    # print(x.shape)

    A = np.pad(x, ((0,0),)+((0, 25-y.shape[0]),)*2, "constant")
    X = np.pad(y, ((0,25-y.shape[0]),(0,0)), "constant")

    assert A.shape == (14,25,25) and X.shape == (25,75)

    return A, X

fingerprint = dc.feat.CircularFingerprint(size=2048, radius=6)

hash_featurizer = dc.feat.CircularFingerprint(size=512, radius=29)

substructures = LoadPickle('Substructures_v5.pickle')[1].values()


def is_substucture(struct, subStruct, count=False):#matches finds if substructure is in structure
    if count:
        return len(struct.GetSubstructMatches(subStruct))
    else:
        return len(struct.GetSubstructMatches(subStruct)) > 0


def find_substuctures(struct, subStructs:List, count=False):#matches list of structure  to structureS
    if count:
        return [is_substucture(struct, ss, count) for ss in subStructs]
    else:
        return [1 if is_substucture(struct, ss, count) else 0 for ss in subStructs]

# def struct_featurizer(state:Chem.Mol) -> np.ndarray:
#         x = featurizers(state).reshape((2048,))
#         return x

def struct_hash(state:Chem.Mol) -> np.ndarray:
    x = hash_featurizer.featurize(state).reshape((512,))
    return tuple(x)

def DrawMol(mol, subfolder, cas, no):
    mol = copy.deepcopy(mol)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    IPythonConsole.drawOptions.addAtomIndices = False
    IPythonConsole.drawOptions.addStereoAnnotation = False
    Draw.MolToFile(Chem.MolFromSmiles(Chem.MolToSmiles(mol)), filename=os.path.join(os.path.abspath(os.path.join('Results',subfolder)),'{}_{}.png'.format(cas,no)))

def DrawSpec(spec, subfolder, cas, no):
    plt.plot(spec)
    plt.savefig(os.path.join(os.path.abspath(os.path.join('Results',subfolder)),'{}_{}.png'.format(cas,no)))
    plt.clf()



####################################### heuristics #######################################

def get_molecular_weight(struct:Chem.Mol):
        return Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(Chem.MolToSmiles(struct)))

def molecular_weight(struct:Chem.Mol, truth:Chem.Mol):
    struct_weight = get_molecular_weight(struct)
    truth_weight = get_molecular_weight(truth)

    return abs(struct_weight-truth_weight)/truth_weight

def jaccard_fingerprint(struct:Chem.Mol, truth:Chem.Mol):
        return jaccard_score(tuple(fingerprint(struct).reshape((2048,))), tuple(fingerprint(truth).reshape((2048,))))

def h_substruct(struct:Chem.Mol, truth:Chem.Mol):
    return int(is_substucture(truth, struct, count=False))

def h_equal(struct:Chem.Mol, truth:Chem.Mol):
    return int(is_substucture(truth, struct, count=False) and len(struct.GetAtoms())==len(truth.GetAtoms()))

def dice(struct:Chem.Mol, truth:Chem.Mol):
    s = find_substuctures(struct, substructures, count=True)
    t = find_substuctures(truth, substructures, count=True)
    
    c = sum([min(a,b) > 0 for a,b in zip(s,t)])
    d = sum([a>0 for a in s])+sum([b>0 for b in t])
    if d==0:
        if c==0:
            return 1
        else:
            return 0
    else:
        return (2*c/d)

def count_substructs(struct:Chem.Mol, truth:Chem.Mol):
    s = find_substuctures(struct, substructures, count=True)
    t = find_substuctures(truth, substructures, count=True)
    
    c = sum([abs(a-b) for a,b in zip(s,t)])
    d = sum(s)+sum(t)
    if d==0:
        if c==0:
            return 1
        else:
            return 0
    else:
        return 1-(c/d)

def ab_pairs(struct:Chem.Mol, truth:Chem.Mol):
    s_ab = {}
    t_ab = {}

    for b in struct.GetBonds():
        bt = b.GetBondType()
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()

        k = str(bt)+str(a1.GetAtomicNum())
        if k not in s_ab:
            s_ab[k] = 1
        else:
            s_ab[k] += 1

        k = str(bt)+str(a2.GetAtomicNum())
        if k not in s_ab:
            s_ab[k] = 1
        else:
            s_ab[k] += 1

    for b in truth.GetBonds():
        bt = b.GetBondType()
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()

        k = str(bt)+str(a1.GetAtomicNum())
        if k not in t_ab:
            t_ab[k] = 1
        else:
            t_ab[k] += 1

        k = str(bt)+str(a2.GetAtomicNum())
        if k not in t_ab:
            t_ab[k] = 1
        else:
            t_ab[k] += 1
    
    s = set(s_ab.keys()).union(set(t_ab.keys()))

    c=0
    d=0
    for k in s:
        a=0
        b=0
        if k in s_ab:
            a = s_ab[k]
        if k in t_ab:
            b = t_ab[k]
        c+=abs(a-b)
        d+=a+b

    if d==0:
        if c==0:
            return 1
        else:
            return 0
    else:
        return 1-(c/d)

def evaluate_struct(struct:Chem.Mol, truth:Chem.Mol):

    return h_substruct(struct, truth)


####################################### permute_mol #######################################

bonds = {
    Chem.rdchem.BondType.SINGLE:1,
    Chem.rdchem.BondType.DOUBLE:2,
    Chem.rdchem.BondType.TRIPLE:3,
    Chem.rdchem.BondType.AROMATIC:1.5,
}

atoms = {
    "C":4,
    "N":3,
    "O":2,
}

# def permute_mol(base:Chem.Mol):
    
#     new_mols = {}

#     i = len(base.GetAtoms())

#     [a.SetAtomMapNum(j) for j,a in enumerate(base.GetAtoms())]

#     for j,a1 in enumerate(base.GetAtoms()):
#         for a2 in atoms:
#             for b in bonds:
                
#                 ### Check if the new atom-bond pair would make an invalid mol
#                 a = base.GetAtoms()[j]
#                 exp_val = sum([bonds[b.GetBondType()] for b in a.GetBonds()])
#                 val_rem = atoms[a.GetSymbol()] - exp_val
#                 if bonds[b] > val_rem or bonds[b] > atoms[a2]:
#                     continue
                
#                 ### Copy the base mol
#                 _base = copy.deepcopy(base)
#                 _j = _base.GetAtoms()[j].GetAtomMapNum()

#                 ### Make the new atom
#                 new_atom = Chem.MolFromSmiles(a2)
#                 _a2 = new_atom.GetAtoms()[0]
#                 _a2.SetAtomMapNum(i)

#                 ### Combine the base mol and new atom (no bond yet)
#                 new_mol = Chem.CombineMols(new_atom, _base)

#                 ### Find the indexes of the pair being bonded
#                 _j = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(_j)
#                 _i = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(i)
#                 new_mol = Chem.EditableMol(new_mol)

#                 ### Make new bond
#                 new_mol.AddBond(_j, _i, order=b)
#                 new_mol = new_mol.GetMol()
#                 Chem.SanitizeMol(new_mol)

#                 # ### Remove Implicit.Explicit Hydrogen
#                 # for a in new_mol.GetAtoms():
#                 #     a.SetNumExplicitHs(0)
#                 #     a.SetNoImplicit(True)
#                 [a.SetAtomMapNum(0) for a in new_mol.GetAtoms()]

#                 new_mols[struct_hash(new_mol)]=new_mol
                
#     return list(new_mols.values())


def permute_mol(base:Chem.Mol, n=None):
    
    new_mols = {}

    i = len(base.GetAtoms())

    [a.SetAtomMapNum(j) for j,a in enumerate(base.GetAtoms())]

    perms = [(j, a1, a2, b) for j,a1 in enumerate(base.GetAtoms()) for a2 in atoms for b in bonds]

    np.random.shuffle(perms)

    if n is None:
        n = len(perms)

    while len(new_mols) < n and perms:
        (j, a1, a2, b) = perms.pop()

        ### Check if the new atom-bond pair would make an invalid mol
        a = base.GetAtoms()[j]
        exp_val = sum([bonds[b.GetBondType()] for b in a.GetBonds()])
        val_rem = atoms[a.GetSymbol()] - exp_val
        if bonds[b] > val_rem or bonds[b] > atoms[a2]:
            continue
        
        ### Copy the base mol
        _base = copy.deepcopy(base)
        _j = _base.GetAtoms()[j].GetAtomMapNum()

        ### Make the new atom
        new_atom = Chem.MolFromSmiles(a2)
        _a2 = new_atom.GetAtoms()[0]
        _a2.SetAtomMapNum(i)

        ### Combine the base mol and new atom (no bond yet)
        new_mol = Chem.CombineMols(new_atom, _base)

        ### Find the indexes of the pair being bonded
        _j = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(_j)
        _i = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(i)
        new_mol = Chem.EditableMol(new_mol)

        ### Make new bond
        new_mol.AddBond(_j, _i, order=b)
        new_mol = new_mol.GetMol()
        Chem.SanitizeMol(new_mol)

        # ### Remove Implicit.Explicit Hydrogen
        # for a in new_mol.GetAtoms():
        #     a.SetNumExplicitHs(0)
        #     a.SetNoImplicit(True)
        [a.SetAtomMapNum(0) for a in new_mol.GetAtoms()]

        new_mols[struct_hash(new_mol)]=new_mol
                
    return list(new_mols.values())


def permute_subs(mol, n=None):
    [a.SetAtomMapNum(i) for i,a in enumerate(mol.GetAtoms())];
    new_mol = Chem.EditableMol(mol)

    order = list(range(len(mol.GetAtoms())))
    random.shuffle(order)

    subs = []

    ### Make new bond
    for i in order:
        a = mol.GetAtomWithIdx(i)
        if len(a.GetBonds()) == 1:
            new_mol.RemoveAtom(i)
            new_mol = new_mol.GetMol()
            Chem.SanitizeMol(new_mol)

            subs.append(new_mol)

            if len(subs) == n:
                break

            new_mol = Chem.EditableMol(mol)
            
    return subs


def modify_mol(base:Chem.Mol, actions:List[Tuple[float, float, Tuple[int, int, int]]], n=1):
    
    new_mols = []

    i = len(base.GetAtoms())

    [a.SetAtomMapNum(j) for j,a in enumerate(base.GetAtoms())]

    out_actions = []

    m=0

    for val, stop, (j, a2, b) in actions:
        new_action = (j, a2, b)
        a2 = list(atoms.keys())[a2-1]
        b = list(bonds.keys())[b-1]

        ### Check if the new atom-bond pair would make an invalid mol
        a = base.GetAtoms()[j]
        exp_val = sum([bonds[b.GetBondType()] for b in a.GetBonds()])
        val_rem = atoms[a.GetSymbol()] - exp_val
        if bonds[b] > val_rem or bonds[b] > atoms[a2]:
            continue
        
        ### Copy the base mol
        _base = copy.deepcopy(base)
        _j = _base.GetAtoms()[j].GetAtomMapNum()

        ### Make the new atom
        new_atom = Chem.MolFromSmiles(a2)
        _a2 = new_atom.GetAtoms()[0]
        _a2.SetAtomMapNum(i)

        ### Combine the base mol and new atom (no bond yet)
        new_mol = Chem.CombineMols(new_atom, _base)

        ### Find the indexes of the pair being bonded
        _j = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(_j)
        _i = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(i)
        new_mol = Chem.EditableMol(new_mol)

        ### Make new bond
        new_mol.AddBond(_j, _i, order=b)
        new_mol = new_mol.GetMol()
        Chem.SanitizeMol(new_mol)

        # ### Remove Implicit.Explicit Hydrogen
        # for a in new_mol.GetAtoms():
        #     a.SetNumExplicitHs(0)
        #     a.SetNoImplicit(True)
        [a.SetAtomMapNum(0) for a in new_mol.GetAtoms()]

        new_mols.append(new_mol)

        out_actions.append((val, stop, new_action))

        m+=1
        if n==m: 
            break

    return new_mols, out_actions


####################################### custom representation #######################################

def rep_struct_featurizer(state:Chem.Mol) -> np.ndarray:

    bond_crossref = {
        1.0: 0,
        2.0: 1,
        3.0: 2,
        # 1.5: 3,
    }

    atom_crossref = {
        6: 0,
        7: 1,
        8: 2,
    }

    # mol = featurizer(state)[0]
    A = np.zeros((4,25,25))
    X = np.zeros((25,3))

    for a in state.GetAtoms():
        j = a.GetIdx()
        X[j][atom_crossref[a.GetAtomicNum()]] = 1
        # A[-1][j][j] = 1

        for b in a.GetBonds():
            if b.GetBeginAtomIdx() == a.GetIdx():
                n = b.GetBeginAtom()
            else:
                n = b.GetEndAtom()

            k = n.GetIdx()

            o = bond_crossref[b.GetBondTypeAsDouble()]
            A[o][j][k] = 1
            A[-1][j][k] = 1

    return A, X

# def _rep_is_substructure(struct:np.array, substruct:np.array, m, n, k=[]):

#     l=len(k)+1
#     if l > n:
#         return False

#     perms = [k+[i,] for i in range(m) if i not in k]
    

#     for p in perms:
#         if np.all(struct[p,p] == substruct[:l,:l]):
#             if _rep_is_substructure(struct, substruct, m, p):
#                 return True
#     return False

# def rep_is_substructure(struct:np.array, substruct:np.array):
#     m = np.sum(np.amax(struct, 1))
#     n = np.sum(np.amax(substruct, 1))
#     if n > m or np.sum(substruct) > np.sum(struct):
#         return False

#     return _rep_is_substructure(struct, substruct, m, n)

def rep_struct_hash(struct:np.array):
    A, X = struct
    X = copy.copy(X[:,:3])

    for i in range(5):
        for j in range(X.shape[0]):
             X[:,j:j+1] = np.matmul(A[j], X[:,j:j+1])

    return hash(tuple(sorted(X.reshape(np.prod(X.shape)))))
            


# def rep_permute_mol(base:tuple, n=None):

#     base_A, base_X = base
    
#     new_mols = {}

#     i = len(base.GetAtoms())

#     # [a.SetAtomMapNum(j) for j,a in enumerate(base.GetAtoms())]

#     m = np.sum(np.amax(base, 1))

#     perms = [(j, a2, b) for j in range(m) for a2 in [0,1,2] for b in [0,2,3]]

#     np.random.shuffle(perms)

#     if n is None:
#         n = len(perms)

#     while len(new_mols) < n and perms:
#         (j, a2, b) = perms.pop()

#         h = [0]*(m*(m-1)/2)

#         ### Check if the new atom-bond pair would make an invalid mol
#         a = base.GetAtoms()[j]
#         exp_val = sum([bonds[b.GetBondType()] for b in a.GetBonds()])
#         val_rem = atoms[a.GetSymbol()] - exp_val
#         if bonds[b] > val_rem or bonds[b] > atoms[a2]:
#             continue
        
#         ### Copy the base mol
#         _base = copy.deepcopy(base)
#         _j = _base.GetAtoms()[j].GetAtomMapNum()

#         ### Make the new atom
#         new_atom = Chem.MolFromSmiles(a2)
#         _a2 = new_atom.GetAtoms()[0]
#         _a2.SetAtomMapNum(i)

#         ### Combine the base mol and new atom (no bond yet)
#         new_mol = Chem.CombineMols(new_atom, _base)

#         ### Find the indexes of the pair being bonded
#         _j = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(_j)
#         _i = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(i)
#         new_mol = Chem.EditableMol(new_mol)

#         ### Make new bond
#         new_mol.AddBond(_j, _i, order=b)
#         new_mol = new_mol.GetMol()
#         Chem.SanitizeMol(new_mol)

#         # ### Remove Implicit.Explicit Hydrogen
#         # for a in new_mol.GetAtoms():
#         #     a.SetNumExplicitHs(0)
#         #     a.SetNoImplicit(True)
#         [a.SetAtomMapNum(0) for a in new_mol.GetAtoms()]

#         new_mols[struct_hash(new_mol)]=new_mol
                
#     return list(new_mols.values())


# def rep_mod_mol(base:tuple, actions:List[Tuple[int, int, int]]):

#     base_A, base_X = base
    
#     new_mols = {}

#     i = len(base.GetAtoms())

#     # [a.SetAtomMapNum(j) for j,a in enumerate(base.GetAtoms())]

#     m = np.sum(np.amax(base, 1))

#     for (j, a2, b) in actions:

#         ### Check if the new atom-bond pair would make an invalid mol
#         a = base.GetAtoms()[j]
#         exp_val = sum([bonds[b.GetBondType()] for b in a.GetBonds()])
#         val_rem = atoms[a.GetSymbol()] - exp_val
#         if bonds[b] > val_rem or bonds[b] > atoms[a2]:
#             continue
        
#         ### Copy the base mol
#         _base = copy.deepcopy(base)
#         _j = _base.GetAtoms()[j].GetAtomMapNum()

#         ### Make the new atom
#         new_atom = Chem.MolFromSmiles(a2)
#         _a2 = new_atom.GetAtoms()[0]
#         _a2.SetAtomMapNum(i)

#         ### Combine the base mol and new atom (no bond yet)
#         new_mol = Chem.CombineMols(new_atom, _base)

#         ### Find the indexes of the pair being bonded
#         _j = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(_j)
#         _i = [a.GetAtomMapNum() for a in new_mol.GetAtoms()].index(i)
#         new_mol = Chem.EditableMol(new_mol)

#         ### Make new bond
#         new_mol.AddBond(_j, _i, order=b)
#         new_mol = new_mol.GetMol()
#         Chem.SanitizeMol(new_mol)

#         # ### Remove Implicit.Explicit Hydrogen
#         # for a in new_mol.GetAtoms():
#         #     a.SetNumExplicitHs(0)
#         #     a.SetNoImplicit(True)
#         [a.SetAtomMapNum(0) for a in new_mol.GetAtoms()]

#         new_mols[struct_hash(new_mol)]=new_mol
                
#     return list(new_mols.values())