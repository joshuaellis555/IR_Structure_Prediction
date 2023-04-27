#from keras.backend.cntk_backend import reshape
from numpy.core.numeric import cross
import sklearn
#from six import b
# from tensorflow.python.framework.ops import convert_to_tensor
# from GA_Classes import Individual, Genome, Fitness
from Modules.ir_functions import *
from Modules.smile_tools import *
from Modules.ga_classes import *
import Modules.ga_classes as GA_Classes
import multiprocessing as mp
import numpy as np
from scipy import spatial
import sys
import keras

import rdkit.Chem as Chem
import deepchem as dc

from typing import List, Tuple, Dict


''' Hyperparameters'''

state_version = '5abssc.2' # Experiment name

fold = 0

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards

epsilon_alpha_min = 0.1  # Minimum epsilon greedy parameter
epsilon_alpha_max = 1.0  # Maximum epsilon greedy parameter
epsilon_alpha_interval = (
    epsilon_alpha_max - epsilon_alpha_min
)  # Rate at which to reduce chance of random action being taken

batch_size = 1024  # Size of batch taken from replay buffer
super_batch_size = 1024*16
max_steps_per_episode = 50

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(
    learning_rate=10**(-4.0),
    clipnorm=0.001
)

# Number of frames to take random action and observe output
epsilon_random_frames = 128
# Number of frames for exploration
epsilon_alpha_greedy_frames = epsilon_random_frames * 4.0

# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_size = 2048 # max examples per unique compound
max_total_memory_size = 420000 #total examples in replay buffer
target_data_subset_size = max_total_memory_size/1

# Train the model after 4 actions
update_after_actions = 4

# How often to update the target network
save_after_frames = 32
update_target_network = 32
# Using huber loss for stability
loss_function1 = keras.losses.Huber()
# loss_function = keras.losses.MeanSquaredError()

total_target_updates = 300 # update_target_network * total_target_updates == 9600

''' ^^^ Hyperparameters ^^^ '''



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
    







def compare_substructre_list(ss,sl,substructures):
    found = [False for _ in range(len(sl))]
    # from Predict_Structures import FindSubstucturesInSubS
    for i in range(len(sl)):
        found[i]=FindSubstucturesInSubS(substructures[sl[i]],substructures[ss])
    return found
        

def build_hierarchy(substructures):
    hierarchy={i:{s:[substructures[s],[],[]] for s in substructures if len(substructures[s].GetAtoms())==i} for i in range(11)}
    for size in range(1,10):
        print('Build Hierarchy Size:',size)

        sl=[s2 for s2 in hierarchy[size+1]]
        jobs = [[ss] for ss in hierarchy[size]]
        _substructures={s:substructures[s] for s in list(hierarchy[size].keys())+list(hierarchy[size+1].keys())}

        found = Buddy(compare_substructre_list,{i:j for i,j in enumerate(jobs)},[sl,_substructures],verbose=True)

        for i in found:
            fl,ss=found[i],jobs[i][0]
            for f,s2 in zip(fl,sl):
                if f:
                    hierarchy[size][ss][1]+=[s2]
                    hierarchy[size+1][s2][2]+=[ss]
        
    return hierarchy


        










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

    from sklearn.metrics import f1_score, jaccard_score
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

    # exit()

    substructures_to_skip = [k for i,k in enumerate(substructure_keys) if 0 in [sum([samples[s]["substructures"][i] for s in sp]) for sp in splits] ]


    validation_ratio = 0.1
    # state_version = '5absscjacc.2'

    # ab_pairs, jaccard_fingerprint, substruct_sim, count_substructs
    evaluation_weights = [1,0,0,1]

    test_label = test_label+'_'+ID+'_V'+state_version

    i=0
    while os.path.exists(test_label+'-Predictions_'+str(i)+'.csv'):
        i+=1
    iCombinedPath=i
    

    








    replicates=range(0,5)

    #k-fold cross validation sets
    cross_validation_sets=[{} for z in range(total_substructures)]
    
    def make_cvs_sets():
        global cross_validation_sets, total_substructures, samples, sample_keys, substructure_keys, splits, validation_ratio

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
    spec = np.array(X_test)
    train_specTitles = list(s_train + s_validate)
    train_spec = {c:x for c,x in zip(train_specTitles, np.array(X_train + X_validate))}
    train_struct = {c:samples[c]['structure'] for c in train_specTitles}
    train_smiles = {c:samples[c]['SMILES'] for c in train_specTitles}

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

       
   




    

    def create_FTIR_model(spec_shape, genome, ex=1):

        LRAlpha = 10**(genome['LRAlpha'][0])

        pathSpec=[]        
        pathSpec.append(Input(shape=spec_shape))

        # Convolutional layer 1 w/ LeakyReLU activation and max pooling
        pathSpec.append(Conv2D(genome['Conv2Dsize'][0] * ex, (3,3), activation='linear',padding='same')(pathSpec[-1]))
        pathSpec.append(BatchNormalization()(pathSpec[-1]))
        pathSpec.append(LeakyReLU(alpha=LRAlpha)(pathSpec[-1]))
        pathSpec.append(MaxPooling2D(padding='same',strides=(1,2))(pathSpec[-1]))

        # Convolutional layer 1 w/ LeakyReLU activation and max pooling
        pathSpec.append(Conv2D(genome['Conv2Dsize'][1] * ex, (3,3), activation='linear',padding='same')(pathSpec[-1]))
        pathSpec.append(BatchNormalization()(pathSpec[-1]))
        pathSpec.append(LeakyReLU(alpha=LRAlpha)(pathSpec[-1]))
        pathSpec.append(MaxPooling2D(padding='same',strides=(1,2))(pathSpec[-1]))

        # Convolutional layer 1 w/ LeakyReLU activation and max pooling
        pathSpec.append(Conv2D(genome['Conv2Dsize'][2] * ex, (3,3), activation='linear',padding='same')(pathSpec[-1]))
        pathSpec.append(BatchNormalization()(pathSpec[-1]))
        pathSpec.append(LeakyReLU(alpha=LRAlpha)(pathSpec[-1]))
        pathSpec.append(MaxPooling2D(padding='same',strides=(1,2))(pathSpec[-1]))

        # Convolutional layer 1 w/ LeakyReLU activation
        pathSpec.append(Conv2D(genome['Conv2Dsize'][3] * ex, (3,3), activation='linear',padding='valid')(pathSpec[-1]))
        pathSpec.append(BatchNormalization()(pathSpec[-1]))
        pathSpec.append(LeakyReLU(alpha=LRAlpha)(pathSpec[-1]))

        pathSpec.append(Flatten()(pathSpec[-1]))

        pathSpec.append(Dense(genome['DenseSize'][0] * ex, activation='linear')(pathSpec[-1]))
        pathSpec.append(BatchNormalization()(pathSpec[-1]))
        pathSpec.append(LeakyReLU(alpha=LRAlpha)(pathSpec[-1]))

        model = Model(inputs=pathSpec[0], outputs=pathSpec[-1])

        return model


    def create_q_model(rep_shape, spec_rep_shape, optimizer, loss_function, genome):

        spec_dense_size = 512
        struct_rep_size = 512
        
        ### --------------------- Struct Path ---------------------

        input1 = layers.Input(shape=rep_shape)

        path1_layer1 = layers.Dense(struct_rep_size, activation="relu")(input1)
        path1_layer2 = layers.Dropout(0.5)(path1_layer1)

        ### --------------------- Spec Path ---------------------

        input3 = layers.Input(shape=spec_rep_shape)

        path2_layer2 = input3
        path2_layer3 = create_FTIR_model(spec_rep_shape, genome, 3)(path2_layer2)

        ### --------------------- Merge Layers ---------------------

        merge_layer1 = layers.Concatenate()([path1_layer2, path2_layer3])

        merge_layer2 = layers.Dense((struct_rep_size+spec_dense_size), activation="relu")(merge_layer1)
        merge_layer3 = layers.Dropout(0.5)(merge_layer2)

        merge_layer4 = layers.Dense((struct_rep_size+spec_dense_size), activation="relu")(merge_layer3)
        merge_layer5 = layers.Dropout(0.5)(merge_layer4)

        merge_layer6 = layers.Dense((struct_rep_size+spec_dense_size), activation="relu")(merge_layer5)
        output1 = layers.Dropout(0.5)(merge_layer6)

        # Actions 
        output1 = layers.Dense(2, activation="linear")(output1)

        model = keras.Model(inputs=[input1,input3], outputs=output1)

        model.compile(optimizer=optimizer, loss=loss_function)

        return model

    featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
    
            










    def struct_representation(state:Chem.Mol) -> np.ndarray:
        x = featurizer.featurize(state).reshape((2048,))
        return x







    def jaccard_fingerprint(struct:Chem.Mol, truth:Chem.Mol):
        return jaccard_score(struct_representation(struct), struct_representation(truth))

    def substruct_sim(struct:Chem.Mol, truth:Chem.Mol):
        s = FindSubstuctures(struct, substructures.values())
        t = FindSubstuctures(truth, substructures.values())
        return 1 - spatial.distance.cosine(s,t)

    def count_substructs(struct:Chem.Mol, truth:Chem.Mol):
        s = FindSubstuctures(struct, substructures.values(), count=True)
        t = FindSubstuctures(truth, substructures.values(), count=True)
        
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

    def evaluate_struct(struct:Chem.Mol, truth:Chem.Mol, hierarchy):
        global evaluation_weights
        weights = evaluation_weights

        comparison = [
            ab_pairs(struct, truth),
            jaccard_fingerprint(struct, truth),
            0,
            count_substructs(struct, truth),
        ]

        return sum([w*c for w,c in zip(comparison, weights)])/sum(weights)


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

    

    def choose_next_state(state:Chem.Mol, model:keras.Model, model_input_shapes:Tuple[tuple,tuple,tuple], spec_rep:np.ndarray, hierarchy:Dict[int,Dict[str,Chem.Mol]], rand_choice:str, truth:Chem.Mol):

        possible_new_states = permute_mol(state)
        
        cur_rep = struct_representation(state)

        true_size = len(truth.GetAtoms())

        random.seed(time.time())

        if len(possible_new_states)==0:
            stop_value = model([
                tf.reshape(tf.convert_to_tensor([cur_rep], dtype=tf.float32), (1,)+model_input_shapes[0]), 
                tf.reshape(tf.convert_to_tensor([spec_rep], dtype=tf.float32), (1,)+model_input_shapes[1]),
            ], training=False)[0][0]
            
            done = len(state.GetAtoms())/true_size
            finished = True
            print('Final Value(null)', evaluate_struct(state,truth,hierarchy) )
            return copy.deepcopy(state), stop_value, done, finished, cur_rep, cur_rep

        if rand_choice=='alpha':

            new_state = random.choice(possible_new_states)
            new_rep = struct_representation(new_state)
            new_size = len(new_state.GetAtoms())
            new_value = None
            done = new_size/true_size
            
            if 1/len(possible_new_states) > np.random.rand(1)[0] or new_size>30:
                finished=True
            else:
                finished=False

            return new_state, new_value, done, finished, cur_rep, new_rep
        else:
            rand_choice = "gama"


        struct_reps = [struct_representation(new_mol) for new_mol in possible_new_states]

        if rand_choice=='gama':
            q_values = model([
                            tf.reshape(tf.convert_to_tensor(struct_reps, dtype=tf.float32), (len(possible_new_states),)+model_input_shapes[0]),
                            tf.reshape(tf.convert_to_tensor([spec_rep for _ in struct_reps], dtype=tf.float32), (len(possible_new_states),)+model_input_shapes[1]),
            ])

            stop_values = q_values[:,1]
            q_values = q_values[:,0]
            best_index = np.argmax(q_values)
            stop_value = stop_values[best_index]
        else:
            print('Unknown Choice!!!!')
        
        if stop_value >= 1:
            finished=True
            print('Final Value', evaluate_struct(state,truth,hierarchy) )
        else:
            finished=False
        
        new_state = possible_new_states[best_index]
        done = len(new_state.GetAtoms())/true_size
        return new_state, q_values[best_index], done, finished, cur_rep, struct_reps[best_index]



    

    cas = train_specTitles

    # from baselines.common.atari_wrappers import make_atari, wrap_deepmind
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers


    
    epsilon_alpha = epsilon_alpha_max  # Epsilon greedy parameter

    # Experience replay buffers
    state_history = {c:[] for c in cas}
    state_rep_history = {c:[] for c in cas}
    state_next_history = {c:[] for c in cas}
    state_next_rep_history = {c:[] for c in cas}
    rewards_history = {c:[] for c in cas}
    rewards_next_history = {c:[] for c in cas}
    done_history = {c:[] for c in cas}
    weight_history = {c:[] for c in cas}
    loss_history = {}
    avg_reward_history = {}
    last_reward = {c:0 for c in cas}



    episode_reward_history = {c:[] for c in cas}
    running_reward = {c:0 for c in cas}
    episode_count = {c:0 for c in cas}
    frame_count = 0
    universal_frame_count = 0


    reps = {c:tf.expand_dims(train_spec[c],0) for c in cas}


    model_input_shapes = (
        (2048,),
        tuple([int(x) for x in train_spec[cas[0]].shape]),
    )

    print('model_input_shapes',model_input_shapes)

    
    zero_time=time.time()

    true_smile = train_smiles





    

    

    tf.keras.backend.set_floatx('float32')
    
    # if os.path.exists('Predict_structure_Encoder_new.h5'):
    #     print('loading encoder')
    #     transformer = keras.models.load_model('Predict_structure_Encoder.h5')
    #     crosscoder = keras.models.load_model('Predict_structure_Crosscoder_ecfp.h5')
    # else:
    #     exit()

    if os.path.exists('States/Struct_Builder_cur_fileno{}.h5'.format(state_version)):
        f=open('States/Struct_Builder_cur_fileno{}.h5'.format(state_version), 'r')
        file_no = int(f.read().strip())
        f.close()
    else: 
        file_no = 0

    print('file_no',file_no)

    
    # if file_no%2 == 1:
    #     flip=1
    # else:
    #     flip=0
    flip=0
    # if True:
    if not os.path.exists('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, (file_no+flip) % 2)):
        # print('new model')
        model = create_q_model(*(model_input_shapes + (optimizer, loss_function1,genome)))
        model_target = create_q_model(*(model_input_shapes + (optimizer, loss_function1,genome)))
    else:
        
        # model = keras.models.load_model('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, (file_no+flip)%2))
        model = create_q_model(*(model_input_shapes + (optimizer, loss_function1,genome)))
        model_target = keras.models.load_model('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, (file_no+flip)%2))
        model.set_weights(model_target.get_weights())

    # print('model done')
        
    # if True:
    if not os.path.exists('States/Struct_Builder_Training_State_{}_{}.state'.format(state_version, file_no % 2)):
        # base_values = {c:evaluate_struct(true_structs[c],train_probabilities[c],hierarchy,None) for c in cas}

        episode_cas = random.sample([c for c in cas], int(0.1 * len(cas)))
        states = {c:Chem.MolFromSmiles("C") for c in episode_cas}
        episode_reward = {c:0 for c in episode_cas}
        # episode_predictions = {c:tf.expand_dims(randomize_predictions(predictions[c]),0) for c in cas}
    else:
        # state_rep_history, state_next_rep_history, rewards_history, done_history, episode_reward_history, running_reward, episode_count, frame_count, base_values, episode_cas, states, episode_reward\
        #     = LoadPickle('States/Struct_Builder_Training_State_{}_{}.state'.format(state_version, file_no % 2))
        # print('loading')

        state_rep_history, state_next_rep_history, state_history, state_next_history, rewards_history, rewards_next_history, done_history, episode_reward_history, running_reward, episode_count, frame_count, episode_cas, states, episode_reward, weight_history, loss_history, last_reward, avg_reward_history\
            = LoadPickle('States/Struct_Builder_Training_State_{}_{}.state'.format(state_version, file_no % 2))

        # episode_predictions = {c:[np.array(e) for e in episode_predictions[c]] for c in cas}

        # frame_count = 0

        # # print('loaded')
        # print(np.array(state_rep_history[cas[0]]).shape)
        # for c in cas:
        #     # state_rep_history[c] = [s.numpy() for s in state_rep_history[c]]
        #     # print(c)
        #     state_rep_history[c] = np.concatenate([state_rep_history[c],crosscoder.predict(np.array(state_rep_history[c]))],axis=-1)
        #     # state_next_rep_history[c] = [s.numpy() for s in state_next_rep_history[c]]
        #     state_next_rep_history[c] = np.concatenate([state_next_rep_history[c],crosscoder.predict(np.array(state_next_rep_history[c]))],axis=-1)

        # print('converted')


        # for c in cas:
        #     episode_count[c]=9

        # episode_cas=[]
        # while len(episode_cas) < int(0.1*len(cas)):
        #     min_episodes = min([len(rewards_history[c]) for c in cas if c not in episode_cas])
        #     episode_cas+= [random.choice( [c for c in cas if len(rewards_history[c])==min_episodes and c not in episode_cas] )]

        # # episode_cas = random.sample([c for c in cas], int(0.1*len(cas)))
        # states = {c:comparisonNode('C',[[0],[0]]) for c in episode_cas}
        # episode_reward={c:0 for c in episode_cas}
        # episode_predictions = {c:tf.expand_dims(tf.convert_to_tensor(randomize_predictions(predictions[c]), dtype=tf.float32),0) for c in cas}

        # state_rep_history = copy.deepcopy(state_next_rep_history)
        # rewards_next_history = copy.deepcopy(rewards_history)

        # print('Updating Reps')
        # for c in cas:
        #     for i in range(len(state_rep_history[c])):
        #         # state_rep_change_zeroed = state_rep_history * np.array([0 if i in range(512,1024) else 1 for i in range(state_rep_history.shape[-1])])
        #         # srzs, unique_indices = np.unique(state_rep_change_zeroed, return_index=True, axis=0)
        #         # for j,s in enumerate(srzs):

        #         # done_history[c] = [sh.size()/true_structs[c].size() for sh in state_history[c]]
        #         state_rep_history[c][i] = struct_representation(state_history[c][i],cross_ref,transformer)
        #         state_next_rep_history[c][i] = struct_representation(state_next_history[c][i],cross_ref,transformer)





    print(episode_cas)

    # base_values = {c:evaluate_struct(true_structs[c],train_probabilities[c],hierarchy,None) for c in cas}
    # done_history = {c:[random.random() for dh in done_history[c]] for c in cas}
    # weight_history = {c:[0.0 for dh in done_history[c]] for c in cas}
    # state_rep_history = {c:[np.reshape(crosscoder(np.reshape(srh, (1,512) )), (512,) ) for srh in state_rep_history[c]] for c in cas}

    # for c in cas:
    #     for i in range(len(rewards_history)):
    #         if np.isnan(rewards_history[c][i]) or np.isnan(rewards_history[c][i]) or np.isnan(rewards_history[c][i]):

    # def fix_representation(rep):

    #     a = rep[0]
    #     c = np.zeros([3+9+27,3+9+27,4])
    #     c[:a.shape[0],:a.shape[1],:] = a

    #     b = rep[1]
    #     d = np.zeros([3+9+27+3+3])
    #     d[:b.shape[0]] = b

        
    #     return (c, d)
    
    # state_rep_history = {c:[fix_representation(srh) for srh in state_rep_history[c]] for c in cas}
    # state_next_rep_history = {c:[fix_representation(snrh) for snrh in state_next_rep_history[c]] for c in cas}

    from tqdm.auto import tqdm
    
    # if False:
    # # if not os.path.exists('States/Struct_Builder_Ground_Truth_{}.state'.format(state_version)):
    #     ground_truth_rep = [struct_representation(true_structs[c],cross_ref,transformer) for c in cas]
    #     ground_truth_struct = [true_structs[c] for c in cas]
    #     ground_truth_next_rep = ground_truth_rep.copy()
    #     ground_truth_next_struct = [true_structs[c] for c in cas]
    #     ground_truth_rewards = [1 for c in cas]
    #     ground_truth_done = [1 for c in cas]
    #     ground_truth_weight = [1 for c in cas]
    #     ground_truth_cas = cas.copy()

    #     todo = [(c,true_structs[c]) for c in cas]
    #     while todo:
            
    #         c,struct=todo.pop()
    #         indx = [i for i,(c2,s) in enumerate(todo) if c2==c and s.size()==struct.size()]
    #         structs = [struct] + [todo[i][1] for i in indx]
    #         todo = [todo[i] for i in range(len(todo)) if i not in indx]

    #         new_structs = []
    #         next_structs = []

    #         for struct in structs:

    #             new_struct = copy.deepcopy(struct)

    #             for i in range(len(new_struct.all_nodes)):
    #                 a=new_struct.all_nodes[i]
    #                 if len(a.bonds)==1:
    #                     new_struct = [na for na in a.bonds[0].links if na.id!=a.id][0]
    #                     a.remove(a.bonds[0],remove_if_empty=False)
    #                     #[n.all_nodes.pop(i) for n in new_struct.all_nodes]
    #                     for n in new_struct.all_nodes.copy():
    #                         n.all_nodes = [n2 for n2 in n.all_nodes if n2.id!=a.id]
    #                     new_structs+=[new_struct.copy(preserve_ID=False)]
    #                     next_structs+=[copy.deepcopy(struct)]
    #                     new_struct = copy.deepcopy(struct)

    #         spec_reps = np.array([struct_representation(n, cross_ref, transformer, zero_loc=True) for n in new_structs])
    #         _, indx = np.unique(spec_reps,return_index=True,axis=0)
    #         spec_reps=[spec_reps[i] for i in indx]
            
    #         print(c,true_structs[c].size(),len(struct.all_nodes),[len(n.all_nodes) for n in new_structs],len(todo),len(spec_reps),len(new_structs),len(ground_truth_cas))
            
    #         new_structs=[new_structs[i] for i in indx]
    #         next_reps=[struct_representation(next_structs[i], cross_ref, transformer, zero_loc=True) for i in indx]
    #         next_structs=[next_structs[i] for i in indx]

    #         ground_truth_cas += [c for _ in new_structs]
    #         ground_truth_struct += [n for n in new_structs]
    #         ground_truth_rep += [r for r in spec_reps]
    #         ground_truth_next_struct += [ns for ns in next_structs]
    #         ground_truth_next_rep += [nr for nr in next_reps]
            
    #         ground_truth_rewards += [evaluate_struct(n,train_probabilities[c],hierarchy,base_values[c]) for n in new_structs]
    #         ground_truth_done += [len(n.all_nodes)/true_structs[c].size() for n in new_structs]
    #         ground_truth_weight += [1 for _ in new_structs]

    #         if np.mean([len(ground_truth_cas),len(ground_truth_struct),len(ground_truth_rep),len(ground_truth_next_struct),len(ground_truth_next_rep),len(ground_truth_rewards),len(ground_truth_done),len(ground_truth_weight)])!=len(ground_truth_cas):
    #             print('Sets Not Equal!!!!!!!!!')
    #             print('len(ground_truth_cas)',len(ground_truth_cas))
    #             print('len(ground_truth_struct)',len(ground_truth_struct))
    #             print('len(ground_truth_rep)',len(ground_truth_rep))
    #             print('len(ground_truth_next_struct)',len(ground_truth_next_struct))
    #             print('len(ground_truth_next_rep)',len(ground_truth_next_rep))
    #             print('len(ground_truth_rewards)',len(ground_truth_rewards))
    #             print('len(ground_truth_done)',len(ground_truth_done))
    #             print('len(ground_truth_weight)',len(ground_truth_weight))

    #         if len(struct.all_nodes)>3:
    #             todo+=[(c,n) for n in new_structs]

    #     SavePickle([
    #         ground_truth_cas,ground_truth_rep,ground_truth_next_rep,ground_truth_rewards,ground_truth_done,ground_truth_weight,
    #     ],'States/Struct_Builder_Ground_Truth_{}.state'.format(state_version))
    # else:
    #     # ground_truth_cas, ground_truth_rep, ground_truth_next_rep, ground_truth_rewards, ground_truth_done, ground_truth_weight = LoadPickle('States/Struct_Builder_Ground_Truth_{}.state'.format(state_version))
    ground_truth_cas, ground_truth_rep, ground_truth_next_rep, ground_truth_rewards, ground_truth_done, ground_truth_weight = [],[],[],[],[],[]

    print('len(ground_truth_cas)',len(ground_truth_cas))



            




    model.summary()
    print('len(episode_cas)',len(episode_cas))
    print('len(state_rep_history)',len(state_rep_history))

    #print('nan in states?:',[np.isnan(s).any() for s in [state_rep_history, state_next_rep_history, rewards_history]])


    # for c in cas:
    #     for i in range(len(state_next_rep_history)):
    #         r = state_next_rep_history[i]
    #         rewards_history[i]=evaluate_struct()

    
    # messy_predictions = {c:{k:(rp,1) for rp in randomize_predictions(predictions[c]) for k in predictions[c]} for c in cas}


    

    lens=[len(state_rep_history[c]) for c in cas]
    print('lens:',min(lens),max(lens),np.mean(lens),np.std(lens),sum(lens),np.mean([episode_count[c] for c in cas]))












    def train_epoch(update_loss_history=False):
        state_rep_sample, state_next_rep_sample, rewards_sample, samples_cas, sample_weights, sample_done=[],[],[],[],[],[]
        # indices = np.random.choice(range(max_memory_size),size=max_memory_size,replace=False)
        
        # for i in indices:
        #     for learned_sample in np.random.choice(cas,len(cas),replace=False):
        #         if i>=len(state_rep_history[learned_sample]):
        #             continue
                
        #         # Using list comprehension to sample from replay buffer
        #         state_rep_sample += [state_rep_history[learned_sample][i]]
        #         state_next_rep_sample += [state_next_rep_history[learned_sample][i]]
        #         rewards_sample += [rewards_history[learned_sample][i]]
        #         samples_cas += [learned_sample]
        #         sample_done += [done_history[learned_sample][i]]
        #         sample_weights += [weight_history[learned_sample][i]]

        #         if len(state_rep_sample)>=target_data_subset_size:
        #             break
        #     if len(state_rep_sample)>=target_data_subset_size:
        #             break

        state_rep_sample = [s for c in cas for s in state_rep_history[c]]
        state_next_rep_sample = [s for c in cas for s in state_rep_history[c]]
        rewards_sample = [s for c in cas for s in rewards_history[c]]
        samples_cas = [c for c in cas for _ in state_rep_history[c]]
        sample_done = [s for c in cas for s in done_history[c]]
        sample_weights = [s for c in cas for s in weight_history[c]]
        
        _ground_truth_rep = np.array(ground_truth_rep, dtype=np.float32)+(\
            (np.random.uniform(-1,1,(len(ground_truth_rep),model_input_shapes[0]))*np.array([1 if i>=512 and i<1024 else 0 for i in range(model_input_shapes[0])]))\
                if len(ground_truth_rep)>0 else 0)
        state_rep_sample = np.array(state_rep_sample, dtype=np.float32)
        #state_rep_sample = np.concatenate([state_rep_sample ,_ground_truth_rep])
        state_next_rep_sample = np.array(state_next_rep_sample + ground_truth_next_rep, dtype=np.float32)
        rewards_sample = np.array(rewards_sample + ground_truth_rewards, dtype=np.float32)
        samples_cas = np.array(samples_cas + ground_truth_cas)
        sample_done = np.array(sample_done + ground_truth_done, dtype=np.float32)
        w=1#np.sum(sample_weights) / len(ground_truth_weight)
        _ground_truth_weight = [gtw*w for gtw in  ground_truth_weight]
        sample_weights = np.array(sample_weights + _ground_truth_weight, dtype=np.float32)

        indices = np.random.choice(np.array(range(len(state_rep_sample))), min(super_batch_size, len(state_rep_sample)))
        #rewards_sample = np.reshape(rewards_sample, (len(rewards_sample),1) )
        # sample_stop = np.array([[1 if sd>=1 else 0] for sd in sample_done])

        print('finished building train set')
        # state_rep_change_zeroed = state_rep_sample * np.array([0 if i in range(512,1024) else 1 for i in range(state_rep_sample.shape[-1])])
        state_rep_n_cas = np.append(state_rep_sample,[[cas.index(c)] for c in samples_cas],axis=-1) #append cas to rep
        unique, unique_inverse = np.unique(state_rep_n_cas[indices], return_inverse=True, axis=0)
        grouped_sets = [[] for _ in unique]
        for i,ui in enumerate(unique_inverse):
            grouped_sets[ui]+=[i]
        

        
        
        


        # def evaluate_local_diversity(sample_done,weights,r=1024):
        #     assd=np.argsort(sample_done)
        #     ret=np.zeros_like(sample_done)
        #     for i in range(len(ret)):
        #         m=i-r
        #         M=i+r
        #         if m<0:
        #             m=0
        #         loc=assd[m:M]
        #         ret[i]=np.std(sample_done[loc])/len(loc)*weights[i]
        #         ret[i]=random.random()*weights[i] if np.isnan(ret[i]) else ret[i]
        #     return ret

        
        # sample_weights = evaluate_local_diversity(sample_done, sample_weights)
        # print(sample_weights,True in np.isnan(sample_weights))
        # sample_weights = sample_weights / np.mean(sample_weights)
        # print(sample_weights,True in np.isnan(sample_weights))
        # print(np.min(sample_weights),np.max(sample_weights),np.mean(sample_weights))



        print('lens:',len(indices),len(state_rep_sample),len(state_next_rep_sample),len(rewards_sample),len(samples_cas),len(sample_weights))
            

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        #print(len(state_next_rep_sample),[s.shape for s in state_next_rep_sample])
        updated_q_values=[]

        #print(sum([np.sum(np.abs(state_next_rep_sample[j]-state_next_rep_sample[0])) for j in range(len(state_next_rep_sample))]))
        # print(np.sum(state_rep_sample, axis=(1,)))

        tf.keras.backend.clear_session()
        for i in range(math.ceil(len(indices)/batch_size)):
            m, M = i*batch_size, min((i+1)*batch_size,len(indices))

            # x = tf.convert_to_tensor(state_next_rep_sample[m:M], dtype=tf.float32)
            # print(x.shape)
            # print((len(samples_cas[m:M]),)+model_input_shapes[0])
            # tf.reshape(x, (len(samples_cas[m:M]),)+model_input_shapes[0])

            # x = tf.convert_to_tensor([reps[sc] for sc in samples_cas[m:M]], dtype=tf.float32)
            # print(x.shape)
            # print((len(samples_cas[m:M]),)+model_input_shapes[1])
            # tf.reshape(x, (len(samples_cas[m:M]),)+model_input_shapes[1])
            
            # print([[type(x) for x in y] for y in state_next_rep_sample[m:M]])
            future_rewards = model_target([
                                        tf.reshape(tf.convert_to_tensor(state_next_rep_sample[m:M], dtype=tf.float32), (len(samples_cas[m:M]),)+model_input_shapes[0]),
                                        tf.reshape(tf.convert_to_tensor([reps[sc] for sc in samples_cas[m:M]], dtype=tf.float32), (len(samples_cas[m:M]),)+model_input_shapes[1]),
            ], training=False)[:,0]

            _updated_q_values = np.reshape(np.array(future_rewards,dtype=np.float32), (M-m,1) )
            if i==0:
                print(_updated_q_values.shape)
            #_updated_q_values = rewards_sample[m:M] + gamma * np.clip(_updated_q_values, a_min=0, a_max=1,)
            updated_q_values += [_updated_q_values]

        updated_q_values = np.concatenate(updated_q_values, axis=0)
        print("updated_q_values.shape,sample_done.shape",updated_q_values.shape,sample_done[indices].shape)
        updated_q_values = np.concatenate([updated_q_values,np.reshape(sample_done[indices],sample_done[indices].shape+(1,))], axis=-1)
            
        # tf.keras.backend.clear_session()
        
        # future_rewards = model_target.predict([tf.reshape(state_next_rep_sample, (len(samples_cas),model_input_shapes[0])),
        #                             tf.reshape([randomize_predictions(predictions[sc]) for sc in samples_cas], (len(samples_cas),model_input_shapes[1])),
        #                             tf.reshape([reps[sc] for sc in samples_cas], (len(samples_cas),model_input_shapes[2])),
        # ], verbose=0, batch_size=batch_size)[:,0]

        # updated_q_values = np.reshape(np.array(future_rewards,dtype=np.float32), (len(future_rewards),1) )
        # updated_q_values = np.concatenate([updated_q_values,np.reshape(sample_done,sample_done.shape+(1,))], axis=-1)

        del future_rewards

        
        tf.keras.backend.clear_session()
        
        
        
        # print('finding final_used_samples')
        # final_used_samples=[]
        # for u in unique_indices:
        #     i=np.argmax(updated_q_values[u])
        #     final_used_samples+=[i]
        # print('len(final_used_samples):',len(final_used_samples))
        # state_rep_sample = state_rep_sample[final_used_samples]
        # state_next_rep_sample = state_next_rep_sample[final_used_samples]
        # rewards_sample = rewards_sample[final_used_samples]
        # samples_cas = samples_cas[final_used_samples]
        # sample_done = sample_done[final_used_samples]
        # sample_weights = sample_weights[final_used_samples]
        # updated_q_values = updated_q_values[final_used_samples]
        # indices = indices[final_used_samples]

        super_indices = list(indices)
        random.shuffle(super_indices)

        # sample_done = sample_done + np.random.normal(0,1/15,size=sample_done.shape[0])
        #updated_q_values = np.concatenate([updated_q_values,np.reshape(sample_done+np.random.normal(0,1/15,size=sample_done.shape[0]),sample_done.shape+(1,))], axis=-1)
        
        for g in grouped_sets:
            updated_q_values[g,0] = np.max(updated_q_values[g,0])
        # print(len(updated_q_values))
        losses = []

        # sample_stop = [int(sd*5) if sd<1 else 5 for sd in sample_done]
        # counts = [sum([1 if ss==v else 0 for ss in sample_stop]) for v in range(12)]
        # weights = sklearn.utils.class_weight.compute_class_weight('balanced',range(6),sample_stop)
        # print(weights)
        # weights = np.array([weights[ss] for ss in sample_stop])
        
        


        all_q_values=[]
        h=0
        while len(super_indices):
            sub_indices, super_indices = super_indices[:batch_size], super_indices[batch_size:]
            with tf.GradientTape() as tape:
                q_indices = [np.argwhere(indices == i)[0][0] for i in sub_indices]
                # print(q_indices, [np.where(indices == i).shape for i in sub_indices])
                # Train the model on the states and updated Q-values
                # q_value = model([state_rep_sample,predictions[learned_sample],reps[learned_sample]])
                # print('Calculating q_values',(i,m,M),upper_bound-1)
                # print(state_rep_sample[sub_indices])
                q_values = model([
                                tf.reshape(tf.convert_to_tensor(state_rep_sample[sub_indices], dtype=tf.float32), (len(sub_indices),)+model_input_shapes[0]),
                                tf.reshape(tf.convert_to_tensor([reps[sc] for sc in samples_cas[sub_indices]], dtype=tf.float32), (len(sub_indices),)+model_input_shapes[1]),
                ], training=True)

                # # Apply the masks to the Q-values to get the Q-value for action taken
                # q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value

                # print(updated_q_values[sub_indices,0].shape)
                # print(q_values[:,0].shape)
                # print(np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,).shape)
                # print((gamma * np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,)).shape)
                # print((rewards_sample[sub_indices] + gamma * np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,)).shape)
                # print(updated_q_values[sub_indices,0].shape)
                x = rewards_sample[sub_indices] + gamma * np.clip(updated_q_values[q_indices,0]-q_values[:,0], a_min=0, a_max=None,)
                # print("!!!x.shape!!!", x.shape, x)
                updated_q_values[q_indices,0] = x

                # print(updated_q_values[sub_indices].shape, sample_stop[sub_indices].shape, q_values[0].shape, q_values[1].shape)

                loss = loss_function1(updated_q_values[q_indices], q_values, sample_weight=sample_weights[sub_indices])
                # loss=(loss1+loss2)/2
                if h == 0:
                    print(updated_q_values.shape)
                    # print('loss:',loss)#,'loss1:',loss1,'loss2:',loss2)
                    print(updated_q_values[q_indices])
                    print(q_values.numpy())
                    print(np.mean(updated_q_values[q_indices,0]-rewards_sample[sub_indices]))
                    print(sample_done[sub_indices])
                    print(q_values[:,1].numpy())
                    print(np.mean(sample_done[sub_indices]-q_values[:,1].numpy()))
                    
            h+=1
            # Backpropagation
            # grads = tape.gradient(loss1, model.trainable_variables)
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # grads = tape.gradient(loss2, model.trainable_variables)
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses+=[loss.numpy()]

        # X=[
        #                 tf.reshape(state_rep_sample, (len(samples_cas),model_input_shapes[0])),
        #                 tf.reshape([randomize_predictions(predictions[sc]) for sc in samples_cas], (len(samples_cas),model_input_shapes[1])),
        #                 tf.reshape([reps[sc] for sc in samples_cas], (len(samples_cas),model_input_shapes[2])),
        # ]

        # q_values = model(X)

        # updated_q_values[:,0] = rewards_sample + gamma * np.clip(updated_q_values[:,0]-q_values[:,0], a_min=0, a_max=None,)

        tf.keras.backend.clear_session()

        # # Apply the masks to the Q-values to get the Q-value for action taken
        # q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value

        # print(updated_q_values[sub_indices,0].shape)
        # print(q_values[:,0].shape)
        # print(np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,).shape)
        # print((gamma * np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,)).shape)
        # print((rewards_sample[sub_indices] + gamma * np.clip(updated_q_values[sub_indices,0]-q_values[:,0], a_min=0, a_max=None,)).shape)
        # print(updated_q_values[sub_indices,0].shape)

        

        # h = model.fit(X, updated_q_values, batch_size=batch_size,epochs=1,verbose=1,shuffle=True)
        # losses+=[h.history['loss'][-1]]

        tf.keras.backend.clear_session()

        # print(h)
                
        


        print('loss (mean, std), losses:', (np.mean(losses), np.std(losses)), losses)
        print('updated_q_values (mean, std):', (np.mean(updated_q_values), np.std(updated_q_values)) )
        print('q_values (mean, std):', (np.mean(q_values), np.std(q_values)) )
        if update_loss_history:
            loss_history[frame_count]=np.mean(losses)


    





    ### ======================================================================= Retrain_Model ===================================================================================== ###
    retrain_model=False
    # retrain_model=True
    if retrain_model:
        print('Re-Updating model')

        for epoch in range(1+8*13):
            print('epoch',epoch+1)
                
            train_epoch()

            if epoch % 8 == 0:
                # update the the target network with new weights
                print('Update target model')
                model_target.set_weights(model.get_weights())
                
                if epoch!=0:
                    print("Saving Model: {}  {}".format(state_version, file_no%2))
                    model.save('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, file_no%2))











    ### ======================================================================= Q-Learning ===================================================================================== ###

    while True:  # Run until solved
        frame_count+=1

        for ic,learned_sample in tqdm(enumerate(episode_cas[:])):

            ### Decay probability of taking random action
            if frame_count > epsilon_random_frames:                
                epsilon_alpha = epsilon_alpha_interval * (epsilon_alpha_greedy_frames - frame_count) / (epsilon_alpha_greedy_frames - epsilon_random_frames) + epsilon_alpha_min
                epsilon_alpha = max(epsilon_alpha, epsilon_alpha_min)

            ### Use epsilon-greedy for exploration
            r = np.random.rand(1)[0]
            if frame_count < epsilon_random_frames or epsilon_alpha > r:
                #Take random action
                rand_choice = 'alpha'
            else:
                rand_choice = 'gama'
            
            tf.keras.backend.clear_session()

            state_next, _, done, finished, cur_rep, next_rep = choose_next_state(
                states[learned_sample], model, model_input_shapes,
                train_spec[learned_sample], hierarchy, rand_choice, train_struct[learned_sample]
            )

            reward = evaluate_struct(states[learned_sample],train_struct[learned_sample],hierarchy)
            reward_next = evaluate_struct(state_next,train_struct[learned_sample],hierarchy)

            if finished == True:
                last_reward[learned_sample] = reward_next

            episode_reward[learned_sample] += reward

            # Save actions and states in replay buffer
            state_history[learned_sample].append(states[learned_sample])
            state_next_history[learned_sample].append(state_next)

            assert cur_rep.shape == (2048,)
            state_rep_history[learned_sample].append(cur_rep)
            state_next_rep_history[learned_sample].append(next_rep)
            done_history[learned_sample].append(done)
            weight_history[learned_sample].append(1)
            rewards_history[learned_sample].append(reward)
            rewards_next_history[learned_sample].append(reward_next)

            states[learned_sample] = state_next

            # Limit the state and reward history
            if np.mean([len(state_rep_history[learned_sample]),
                    len(state_next_rep_history[learned_sample]),
                    len(state_history[learned_sample]),
                    len(state_next_history[learned_sample]),
                    len(done_history[learned_sample]),
                    len(weight_history[learned_sample]),
                    len(rewards_history[learned_sample]),
                    len(rewards_next_history[learned_sample]),
            ]) != len(state_rep_history[learned_sample]):
                print('Lengths Are Not Equal!!!!!')
                print([len(state_rep_history[learned_sample]),
                        len(state_next_rep_history[learned_sample]),
                        len(state_history[learned_sample]),
                        len(state_next_history[learned_sample]),
                        len(done_history[learned_sample]),
                        len(weight_history[learned_sample]),
                        len(rewards_history[learned_sample]),
                        len(rewards_next_history[learned_sample]),
                ])
                exit()




            if len(rewards_history[learned_sample]) > max_memory_size:
                # print("Deleting old history!!!")
                del rewards_history[learned_sample][:-max_memory_size]
                del rewards_next_history[learned_sample][:-max_memory_size]

                del state_history[learned_sample][:-max_memory_size]
                del state_next_history[learned_sample][:-max_memory_size]

                del state_rep_history[learned_sample][:-max_memory_size]
                del state_next_rep_history[learned_sample][:-max_memory_size]
                
                del done_history[learned_sample][:-max_memory_size]
                del weight_history[learned_sample][:-max_memory_size]
                

            # Update running reward to check condition for solving
            episode_reward_history[learned_sample].append(episode_reward[learned_sample])
            if len(episode_reward_history[learned_sample]) > 100:
                del episode_reward_history[learned_sample][:1]
            running_reward[learned_sample] = np.mean(episode_reward_history[learned_sample])

            # Log details
            template = "cas:{} [{}] running reward: {:.2f} at episode {}, frame count {}, done {}"
            

            if finished:  # Condition to consider the task solved
                episode_count[learned_sample] += 1

                lens = {c:len(rewards_history[c]) for c in cas}
                print('\tcas:',learned_sample,' True SMILES:', true_smile[learned_sample],' Predicted SMILES:', Chem.MolToSmiles(states[learned_sample]),' reward:',rewards_history[learned_sample][-1])

                min_episodes = min([len(rewards_history[c]) for c in cas if c not in episode_cas])
                episode_cas[ic] = random.choice( [c for c in cas if len(rewards_history[c])==min_episodes and c not in episode_cas] )

                learned_sample = episode_cas[ic]
                
                states[learned_sample] = Chem.MolFromSmiles("C")
                episode_reward[learned_sample] = 0
                continue
        
        lens = [len(rewards_history[c]) for c in cas]
        while sum(lens) > max_total_memory_size:
            max_len=max(lens)
            indices=[i for i in range(len(cas)) if lens[i]==max_len]
            i = np.random.choice(indices,1)[0]
            c=cas[i]
            # print("Deleting old history!!!")
            del rewards_history[c][:1]
            del rewards_next_history[c][:1]

            del state_history[c][:1]
            del state_next_history[c][:1]

            del state_rep_history[c][:1]
            del state_next_rep_history[c][:1]

            del done_history[c][:1]
            del weight_history[c][:1]
            lens = [len(rewards_history[c]) for c in cas] #update lens

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and sum([len(done_history[c]) for c in cas]) >= batch_size:

            print('Updating model')
            
            train_epoch(update_loss_history=True)
            avg_reward_history[frame_count] = np.mean([last_reward[c] for c in cas])

        ### Finish Updating Model


        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            print('Update target model')
            model_target.set_weights(model.get_weights())

        print('Rewards:',np.mean([last_reward[c] for c in episode_cas])
                ,'epsilon_alpha:',epsilon_alpha
                ,'episode_count',np.mean([episode_count[c] for c in cas]),'frame_count', frame_count, 'save in:', save_after_frames - frame_count % save_after_frames
                ,'avg size:',np.mean([len(states[c].GetAtoms()) for c in episode_cas if c in states]),
        )

        if frame_count % save_after_frames == 0:
            
            file_no += 1
            print("Saving Model and State: {}  {}".format(state_version, file_no % 2))

            model.save('States/Struct_Builder_Model_{}_{}.h5'.format(state_version, file_no%2))
            SavePickle((
                state_rep_history, state_next_rep_history, state_history, state_next_history, rewards_history, rewards_next_history, done_history, episode_reward_history, running_reward, episode_count, frame_count, episode_cas, states, episode_reward, weight_history, loss_history, last_reward, avg_reward_history
            ),'States/Struct_Builder_Training_State_{}_{}.state'.format(state_version, file_no % 2))

            f=open('States/Struct_Builder_cur_fileno{}.h5'.format(state_version), 'w+')
            f.write(str(file_no))
            f.close()
            
            if file_no >= total_target_updates:
                print(f"Stopping at file_no {file_no}")
                exit()