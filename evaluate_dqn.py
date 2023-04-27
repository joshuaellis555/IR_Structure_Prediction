from cmath import nan
from tensorflow.python.ops.gen_math_ops import sub
from GA_Classes import Individual, Genome, Fitness
from Modules.ir_functions import *
from Modules.smile_tools import *
from GA_Classes import *
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA

from scipy import spatial

import rdkit.Chem as Chem
from rdkit.Chem import Draw
import deepchem as dc

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = False
IPythonConsole.drawOptions.addStereoAnnotation = False

''' Experiment name and state name '''

state_version = '5abssc.2' # Experiment name
version = 3 # Simple identifier for results, can be any number

fold = 0 # Be sure this is the same as the trained model

''' Experiment setup '''

# weights should be same metrics used in loaded state
evaluation_weights = [
    1, # (abc) Atom Bond Counts
    0, # (jacc) Jaccard Similarity of EFCP
    0, # 'ssim' substructure sim
    1, # (ssc) Substructure Counts
    0, # 'molw' similarity of molecular weights
    0, # 'ex_match' is it an exact match
]

''' ^^^ ^^^ '''


def DrawMol(mol, subfolder, cas, no):
    Draw.MolToFile(Chem.MolFromSmiles(Chem.MolToSmiles(mol)), filename=os.path.join(os.path.abspath(os.path.join('Results',subfolder)),'{}_{}.png'.format(cas,no)))

CircularFingerprint = dc.feat.CircularFingerprint(size=2048, radius=4)

def Featurize(s,featurizer):
    return featurizer.featurize(s)

def IsSubstructure(struct,substruct):
    return len(struct.GetSubstructMatches(substruct)) > 0

def FindSubstucturesInSubS(struct, subStruct, count=False):#matches substructure to structure
    if count:
        return len(struct.GetSubstructMatches(subStruct))
    else:
        return len(struct.GetSubstructMatches(subStruct)) > 0


def FindSubstuctures(struct, subStructs, count=False):#matches structure to structureS
    if count:
        return [FindSubstucturesInSubS(struct, ss, count) for ss in subStructs]
    else:
        return [1 if FindSubstucturesInSubS(struct, ss, count) else 0 for ss in subStructs]


class mem_state():
    def __init__(self, **args):
        self.__dict__ = args
    
    def vars(self):
        return self.__dict__
    
    def vals(self):
        return self.__dict__.copy()

    def merge(self, **args):
        c = self.__dict__.copy()
        c.update(args)
        return mem_state(**c)

    def update(self, **args):
        self.__dict__.update(args)

    def copy(self):
        return copy.copy(self)
    
    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.__iter__()

    def __getitem__(self, __name: str):
        return self.__dict__[__name]

    def __setitem__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value




            






if __name__ == '__main__':
    #from __future__ import print_function
    import os
    import sys
    import random
    import numpy as np
    import math
    from Modules.mp_tools import Buddy
    from functools import partial
    import time
    import copy

    import tensorflow.keras as keras
    from tensorflow.keras.models import Model,Sequential,load_model
    from tensorflow.keras.layers import Input,Concatenate,Dense, Dropout, Flatten, BatchNormalization, Activation
    from tensorflow.keras.layers import SeparableConv1D, MaxPooling1D,MaxPooling2D,Reshape,Conv1D,Conv2D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.constraints import max_norm
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
    from tensorflow.keras.activations import softmax
    from tensorflow.keras.callbacks import Callback

    from sklearn.metrics import f1_score, jaccard_score, fbeta_score
    import tensorflow as tf
    import tensorflow.keras.backend as K

    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #'''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('gpus!!!!!!',gpus)
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    #'''

    comparisonStructures=[]

    Atoms={'H':1,'C':4,'O':2,'N':3,'F':1,'Cl':1,'Br':1,'I':1,'Si':4, 'P':5, 'S':6, 'O-':1, 'N-':2,'?':6,'N+':4}
    Bonds={'-':1, '=':2, '#':3}

    class_fname='classifiedStructsAll_v5.classified'

    #####test_label='MetamorphicResults_v5_Epochs200_Full' ##### <--------- TEST NAME <--------- #####
    test_label='PredictStructures' ##### <--------- TEST NAME <--------- #####

    seed=1234 #<--SET SORTING SEED HERE
    random.seed(seed)
    print('seed: ',seed)


    replicates=range(0,5)
    # replicates=range(-1,0)


    test_label='Substructures_Basic_wH' ##### <--------- TEST NAME <--------- #####

    ID = str(random.randrange(sys.maxsize))
    print('ID:',ID)


    sample_keys, samples = LoadPickle('Data/UniqueSamples_NoRings_v5.pickle')
    total_samples=len(samples)

    substructure_keys, substructures = LoadPickle('Data/Substructures_v5.pickle')
    total_substructures = len(substructure_keys)

    
    split_fname = "Data/DatasetSplit_Min10_v5"
    if not os.path.exists(split_fname+".pickle"):
        splits = LoadPickle(split_fname+".state")["pop"]
        objective_order=[1,0,2]
        for i in objective_order:
            m=min([p.fitness[i] for p in splits])
            splits = [p for p in splits if p.fitness[i]==m]
        splits = splits[0]

        print(splits.fitness)
        splits = splits.genome["set_ref"]
        splits = [[sample_keys[j] for j in range(len(splits)) if splits[j]==i] for i in range(min(splits),max(splits)+1)]

        SavePickle(splits, split_fname+".pickle")
    else:
        splits = LoadPickle(split_fname+".pickle")

    print([len(split) for split in splits])
    for i,s1 in enumerate(splits[:-1]):
        for s2 in splits[i+1:]:
            print(len(set(s1).intersection(set(s2))))

    substructures_to_skip = [k for i,k in enumerate(substructure_keys) if 0 in [sum([samples[s]["substructures"][i] for s in sp]) for sp in splits] ]

    num_predictions = 5
    num_final_predictions = 5

    sim_factor = 1.0

    validation_ratio = 0.1

    test_label = test_label+'_'+ID+'_V'+state_version
    # test_label = test_label+'_'+ID+'_V5'

    experiement_title = 'model-pred_sv{}_np{}_sf{}_v{}'.format(state_version,num_predictions,sim_factor,version)

    i=0
    while os.path.exists(test_label+'-Predictions_'+str(i)+'.csv'):
        i+=1
    iCombinedPath=i
    

    










    #k-fold cross validation sets
    cross_validation_sets=[{} for z in range(total_substructures)]
    
    def make_cvs_sets():
        global cross_validation_sets, total_substructures, samples, sample_keys, substructure_keys, splits, validation_ratio

        
        #trainRatio=trainRatio*data_ratio

        for z in range(total_substructures):
            for k in replicates:
                
                s_train = []
                s_validate = []
                s_test = []

                X_train=[]
                X_validate=[]
                X_test=[]
                
                Y_train=[]
                Y_validate=[]
                Y_test=[]
                

                for i in range(len(splits)-1):
                    for s in splits[(i+k) % len(splits)]:
                        s_train+=[ s ]
                        X_train+=[ samples[s]["processed_spec"] ]
                        Y_train+=[ [samples[s]["substructures"][z]] ]

                t=list(zip(X_train,Y_train,s_train))
                random.shuffle(t)
                positive_t = [_t for _t in t if _t[1][0]==1]
                negative_t = [_t for _t in t if _t[1][0]==0]
                val = [p for i,p in enumerate(positive_t) if i<max(int(len(positive_t)*0.1),1)]+[n for i,n in enumerate(negative_t) if i<max(int(len(negative_t)*0.1),1)]
                train = [p for i,p in enumerate(positive_t) if i>=max(int(len(positive_t)*0.1),1)]+[n for i,n in enumerate(negative_t) if i>=max(int(len(negative_t)*0.1),1)]
                X_train, Y_train, s_train = zip(*train)
                X_validate, Y_validate, s_validate = zip(*val)
                    
                for s in splits[(len(splits)-1+k) % len(splits)]:
                        s_test+=[ s ]
                        X_test+=[ samples[s]["processed_spec"] ]
                        Y_test+=[ [samples[s]["substructures"][z]] ]

                if z==0 and k==0:
                    print('Overlap (train,val):', len(set(s_train).intersection(s_validate)))
                    print('Overlap (test,val):', len(set(s_test).intersection(s_validate)))
                    print('Overlap (train,test):', len(set(s_train).intersection(s_test)))
                    print(len(s_train)+len(s_validate)+len(s_test),
                        len(samples),
                        len(set(samples.keys())),
                        len(set(splits[0]+splits[1]+splits[2]+splits[3]+splits[4])),
                        len(splits[0]+splits[1]+splits[2]+splits[3]+splits[4]),
                        len(splits),
                    )
                
                cross_validation_sets[z][k]=[(X_train,Y_train,s_train),(X_test,Y_test,s_test),(X_validate,Y_validate,s_validate)]

    Results=[[] for i in range(replicates[-1]+1)]

    make_cvs_sets()





    



    
    

    fold=fold
    z=0
    (X_train,Y_train,s_train),(X_test,Y_test,s_test),(X_validate,Y_validate,s_validate) = cross_validation_sets[z][fold]

    print( len(s_train), len(s_test), len(s_validate), len(set(s_train).intersection(set(s_test))), len(set(s_validate).intersection(set(s_train))), len(set(s_test).intersection(set(s_validate))) )

    specTitles = list(s_test)
    all_smiles = list(s_test) + list(s_train) + list(s_validate)
    print(False in ['C' in s for s in all_smiles])
    all_smiles = sorted(all_smiles, key= lambda x: len(x.strip()))
    f = open("all_SMILES.txt", "w")
    f.write('\n'.join(all_smiles))
    f.close()
    # exit()

    spec = {c:x for c,x in zip(specTitles, np.array(X_test))}
    true_structs = {c:samples[c]['structure'] for c in specTitles}
    true_smile = {c:samples[c]['SMILES'] for c in specTitles}

    # Results={s:{} for s in sample_keys}
    # usedClasses=[]
    # test_outputs=[[] for s in specTitles]
    # train_outputs=[[] for s in train_specTitles]
    # f_scores = {c:0 for c in substructure_keys}

    classes = {s:cross_validation_sets[z][fold][1][1] for z,s in enumerate(substructure_keys)}
    train_classes = {s:cross_validation_sets[z][fold][0][1]+cross_validation_sets[z][fold][2][1] for z,s in enumerate(substructure_keys)}

    class_counts = {s:np.sum(classes[s]) for s in substructure_keys}
    train_counts = {s:np.sum(train_classes[s]) for s in substructure_keys}
    total_counts = {s:class_counts[s]+train_counts[s] for s in substructure_keys}

    State=LoadPickle('Data/GA_v4_No4F1_HyperparameterCurves.pickle') #Hyperparameters to load

    def domination(pop,rank=0,shuffle=True):
        dominated=[]
        non_dominated=[]
        for ind in pop:
            is_dominated=False
            for p in pop:
                if ind.fitness>p.fitness:
                    dominated+=[ind]
                    is_dominated=True
                    break
            if not is_dominated:
                non_dominated+=[ind]
        #print('DOM:',len(dominated),len(non_dominated))
        for p in non_dominated: p.rank=rank
        if shuffle:
            random.shuffle(non_dominated)
        else:
            non_dominated.sort(key=lambda x:x.fitness._fitness)
        if len(dominated)>0:
            return non_dominated+domination(dominated,rank=rank+1)
        else:
            return non_dominated

    dominant=domination(State['pop_history'][-1],shuffle=False)
    genome=dominant[0].genome

    hierarchy = None

    featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)










   
   


    def struct_representation(state:Chem.Mol) -> np.ndarray:
        return featurizer.featurize(state).reshape((2048,))

    def jaccard_fingerprint(struct:Chem.Mol, truth:Chem.Mol):
        return jaccard_score(struct_representation(struct), struct_representation(truth))

    def substruct_sim(struct:Chem.Mol, truth:Chem.Mol):
        s = FindSubstuctures(struct, substructures.values())
        t = FindSubstuctures(truth, substructures.values())
        return 1 - spatial.distance.cosine(s,t)

    def count_substructs(struct:Chem.Mol, truth:Chem.Mol):
        s = FindSubstuctures(struct, substructures.values(), count=True)
        t = FindSubstuctures(truth, substructures.values(), count=True)
        # print(len(substructures), substructures.keys())
        
        c = sum([abs(a-b) for a,b in zip(s,t)])
        d = sum(s)+sum(t)
        if d==0:
            if c==0:
                return 1
            else:
                return 0
        else:
            return 1-(c/d)

    def element_ratio(struct:Chem.Mol, truth:Chem.Mol, element:str):
        atom = Chem.MolFromSmiles(element)
        atom = Chem.rdmolops.RemoveAllHs(atom)
        s = FindSubstucturesInSubS(struct, atom, count=True)
        t = FindSubstucturesInSubS(truth, atom, count=True)
        
        return s/t if t!=0 else np.nan

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


    def get_molecular_weight(struct:Chem.Mol):
        return Chem.rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(Chem.MolToSmiles(struct)))

    def molecular_weight(struct:Chem.Mol, truth:Chem.Mol):
        struct_weight = get_molecular_weight(struct)
        truth_weight = get_molecular_weight(truth)

        return abs(struct_weight-truth_weight)/truth_weight

    def mol_size(struct:Chem.Mol) -> int:
        return len(struct.GetAtoms())

    def exact_match(struct:Chem.Mol, truth:Chem.Mol):
        return 1 if FindSubstucturesInSubS(truth, struct) and len(struct.GetAtoms()) == len(truth.GetAtoms()) else 0 


    def evaluate_struct(struct:Chem.Mol, truth:Chem.Mol, hierarchy):
        global evaluation_weights
        mem = mem_state()
        mem.update(
            ab = ab_pairs(struct, truth),
            jacc = jaccard_fingerprint(struct, truth),
            ssim = substruct_sim(struct, truth),
            ssc = count_substructs(struct, truth),
            molw_dev = molecular_weight(struct, truth),
            ex_match = exact_match(struct, truth),
            C_ratio = element_ratio(struct, truth, 'C'),
            N_ratio = element_ratio(struct, truth, 'N'),
            O_ratio = element_ratio(struct, truth, 'O'),
            molw = get_molecular_weight(struct),
            mols = mol_size(struct),
        )
        
        if 0 in (mem.ab, mem.jacc, mem.ssim):
            mem.update(old = 0.0)
        else:
            mem.update(old = 3/sum([1/(c) for c in [mem.ab, mem.jacc, mem.ssc]]))
            
        
        val = sum([mem[k]*evaluation_weights[i] for i,k in enumerate(['ab', 'jacc', 'ssim', 'ssc', 'molw', 'ex_match'])]) / sum(evaluation_weights)
        mem.update(val = val)

        return mem

        
    





    import copy
    import itertools
    import time

    bonds = {
        Chem.rdchem.BondType.SINGLE:1,
        Chem.rdchem.BondType.DOUBLE:2,
        Chem.rdchem.BondType.TRIPLE:3,
        # Chem.rdchem.BondType.AROMATIC,
    }

    atoms = {
        "C":4,
        "O":2,
        "N":3,
    }

    def permute_mol(base:Chem.Mol):
    
        new_mols = []

        i = len(base.GetAtoms())

        [a.SetAtomMapNum(i) for i,a in enumerate(base.GetAtoms())]

        for j,a1 in enumerate(base.GetAtoms()):
            for a2 in atoms:
                for b in bonds:
                    
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

                    new_mols.append(new_mol)
                    
        return new_mols









    import copy
    import itertools
    import time

    def find_next_states(state, model, model_input_shapes, spec_rep):
        possible_new_states = permute_mol(state)
        
        cur_rep = struct_representation(state)

        if len(possible_new_states)==0:
            return [],[],[],[]

        else:
            struct_reps = [struct_representation(new_mol) for new_mol in possible_new_states]

            q_values = model([
                            tf.reshape(tf.convert_to_tensor(struct_reps, dtype=tf.float32), (len(possible_new_states),)+model_input_shapes[0]),
                            tf.reshape(tf.convert_to_tensor([spec_rep for _ in struct_reps], dtype=tf.float32), (len(possible_new_states),)+model_input_shapes[1]),
            ])

            stop_values = q_values[:,1]
            q_values = q_values[:,0]

            return possible_new_states, struct_reps, q_values, stop_values



    




    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    

    

    



    cas = specTitles[:]

    reps = {c:tf.expand_dims(spec[c],0) for c in cas}

    model_input_shapes = (
        (2048,),
        tuple([int(x) for x in spec[cas[0]].shape]),
    )

    print('model_input_shapes',model_input_shapes)

    
    zero_time=time.time()






    

    tf.keras.backend.set_floatx('float32')

    if os.path.exists('States/Struct_Builder_cur_fileno{}.h5'.format(state_version)):
        f=open('States/Struct_Builder_cur_fileno{}.h5'.format(state_version), 'r')
        file_no = int(f.read().strip())
        f.close()
    else: 
        print("file_no not found!")
        exit()

    print('file_no',file_no)

    
    print("Loading Model")


    flip=0
    if not os.path.exists('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, (file_no+flip) % 2)):
        print("Model file not found!")
        exit()
    else:
        model = keras.models.load_model('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, (file_no+flip)%2))

    print("Loaded Model")
        





    

    

    

    os.makedirs('Results/{}'.format(experiement_title), exist_ok=True)

    true_mod_values = {c:model.predict([
                            tf.reshape(tf.convert_to_tensor(struct_representation(true_structs[c]), dtype=tf.float32), (1,)+model_input_shapes[0]),
                            tf.reshape(tf.convert_to_tensor([reps[c]], dtype=tf.float32), (1,)+model_input_shapes[1]),
            ])[0,0] for c in cas}

    optimizer = keras.optimizers.Adam(
        learning_rate=10**(-4.0),
        clipnorm=0.001
    )


       


    class prediction:
        def __init__(self, struct, value, done, rep=[],evaluation=None):
            self.struct = struct
            self.value = value
            self.is_new = True
            self.done = done
            self.fingerprint = struct_representation(struct)
            self.evaluation=evaluation
            self.full_evaluation=None

    def add_unique(unique_pop,new_pop):
        unique_pop = unique_pop[:]
        for new_p in new_pop:
            is_unique = True
            for uni_p in unique_pop:
                if np.sum(np.abs(new_p.fingerprint - uni_p.fingerprint))==0:
                    is_unique = False
                    break
            if is_unique:
                unique_pop+=[new_p]
        return unique_pop

    from functools import cmp_to_key







    folder='Results/{}/'.format(experiement_title)
    ex_results={}
        


    evaluations = {c:[] for c in cas}
    full_evaluations = {c:mem_state(done_accept=0, incomplete_accept=0, done_reject=0, incomplete_reject=0) for c in cas}
    for ic, c in enumerate(cas):

        improved = True
        incomplete_predictions = [prediction(Chem.MolFromSmiles("C"),-np.inf,0)]
        final_predictions = []
        i=0
        print(ic, "of", len(cas), c,'true_smile[c]',true_smile[c],file_no%2)
        
        while improved:
            
            new_predictions = []
            for sp in incomplete_predictions:
                if not sp.is_new:
                    continue
                sp.is_new = False
                # print(sp.struct.size())
                new_states, new_reps, q_values, done = find_next_states(sp.struct, model, model_input_shapes, reps[c])
                new_predictions += [prediction(ns, float(qv), float(d), nr,) for ns,nr,qv,d in zip(new_states, new_reps, q_values, done)]

            incomplete_predictions = add_unique(incomplete_predictions, [n for n in new_predictions if n.done<1])
            final_predictions = add_unique(final_predictions, [n for n in new_predictions if n.done>=1])


            def good_and_unique(pop,n,q):
                pop.copy()
                ret_pop=[]
                # pop, ret_pop = pop[:-1], [pop[-1]]

                def similarity(pop):
                    x=np.concatenate([[p.fingerprint] for p in pop])
                    # print(x,x.shape)
                    x=np.std(x,axis=-1)
                    # print(x)
                    x=np.average(x)
                    # print(x)
                    return x

                while len(ret_pop)<n and len(pop)>0:
                    best_i=0
                    best=-np.inf
                    for i,p in enumerate(pop):
                        s=similarity(ret_pop+[p])
                        v=np.average([_p.value for _p in ret_pop+[p]])
                        x=v+q*s
                        if x>best:
                            best=x
                            best_i=i
                    ret_pop=ret_pop+[pop.pop(best_i)]
                
                return ret_pop
                            
            full_evaluations[c].incomplete_reject += len(incomplete_predictions)
            incomplete_predictions = good_and_unique(incomplete_predictions,num_predictions,sim_factor)
            full_evaluations[c].incomplete_reject -= len(incomplete_predictions)

            full_evaluations[c].done_reject += len(final_predictions)
            final_predictions = good_and_unique(final_predictions,num_predictions,sim_factor)
            full_evaluations[c].done_reject -= len(final_predictions)

            if True in [sp.is_new for sp in incomplete_predictions]:
                improved = True
            else:
                improved = False

        # struct_predictions = good_and_unique(struct_predictions,num_predictions,1)
        if len(final_predictions)<num_final_predictions:
            full_evaluations[c].incomplete_accept += num_final_predictions-len(final_predictions)
            final_predictions += incomplete_predictions[:num_final_predictions-len(final_predictions)]

        final_predictions = add_unique(final_predictions, incomplete_predictions)
        final_predictions = good_and_unique(final_predictions,num_final_predictions,sim_factor)
        full_evaluations[c].done_accept += len(final_predictions)-full_evaluations[c].incomplete_accept

        full_evaluations[c].incomplete_reject -= full_evaluations[c].incomplete_accept
        full_evaluations[c].done_reject -= full_evaluations[c].done_accept

        for i,sp in enumerate(final_predictions):
            print('\t',sp.value,Chem.MolToSmiles(sp.struct))
            evaluation = evaluate_struct(sp.struct, true_structs[c], hierarchy)

            for k in evaluation:
                if k not in full_evaluations[c].keys():
                    full_evaluations[c][k] = [evaluation[k]]
                else:
                    full_evaluations[c][k] += [evaluation[k]]
            
            DrawMol(sp.struct, experiement_title, c, '{}_value{}_truesim{}'.format(i,round(sp.value*100,2),round(evaluation['val']*100,2)))
        DrawMol(true_structs[c], experiement_title, c, '_truth')

    results={c:mem_state() for c in cas}

    for c in cas:
        for k in full_evaluations[c]:
            if isinstance(full_evaluations[c][k],list):
                results[c]['top1_'+k] = full_evaluations[c][k][0]
                if k == 'molw':
                    for i in range(5):
                        results[c][f'{k}_{i}'] = full_evaluations[c][k][i]
                elif k == 'mols':
                    for i in range(5):
                        results[c][f'{k}_{i}'] = full_evaluations[c][k][i]
                elif k == 'molw_dev':
                    results[c]['mean_'+k] = np.mean(full_evaluations[c][k][:5])
                else:
                    results[c]['top5_'+k] = np.max(full_evaluations[c][k][:5])
            else:
                results[c][k] = full_evaluations[c][k]
        
        results[c]['true_molw'] = get_molecular_weight(true_structs[c])
        results[c]['true_size'] = mol_size(true_structs[c])

    f=open(folder+'_full_eval_{}.csv'.format(experiement_title),'w+')
    f.write(','+','.join(results[cas[0]])+'\n')
    for c in cas:
        f.write(c+','+','.join([str(results[c][k]) for k in results[c]])+'\n')
    f.close()