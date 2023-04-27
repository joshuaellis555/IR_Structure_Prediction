import os
import sys
from PIL import Image
from shutil import copyfile
from Modules.mp_tools import Buddy
from Modules.ir_functions import *
import itertools
import numpy as np

Elements=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',
          'Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Ti','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U',
          'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

Weights=[1.008,4.0026,6.94,9.0122,10.81,12.011,14.007,15.999,18.998,20.180,22.990,24.305,26.982,28.085,30.974,32.06,35.45,39.948,39.098,40.078,44.956,47.867,50.942,51.996,54.938,55.845,58.933,58.693,63.546,65.38,69.723,72.630,74.922,
         78.971,79.904,83.798,85.468,87.62,88.906,91.224,92.906,95.95,98,101.07,102.91,106.42,107.87,112.41,114.82,118.71,121.76,127.60,126.90,131.29,132.91,137.33,138.91,140.12,140.91,144.24,145,150.36,151.96,157.25,158.93,162.50,
         164.93,167.26,168.93,173.05,174.97,178.49,180.95,183.84,186.21,190.23,192.22,195.08,196.97,200.59,204.38,207.2,208.98,209,210,222,223,226,227,232.04,231.04,238.03,237,244,243,247,247,251,252,257,258,259,266,267,268,269,170,
         270,278,281,282,285,286,289,290,293,294,294]

Bonds={'-':1,'=':2,'#':3}

class structure():
    '''
    A set of nodes and bonds making up the compound's structure
    '''
    
    def __init__(self,smile,refs=None,ID=None,isinit=True):
        if not refs:
            refs={}#refrences within smile. Denoted by numbers in the smile
        if not ID:
            ID=[[0],[0]]#id counter used for nodes. Gets passed to substructures when they are processed

        base_smile=smile
        smile=list(smile)#convert smile from string to list of characters
        
        i=0 #combine charicters for multi-digit numbers
        numbers_found=[0]
        while i<len(smile)-1:
            if smile[i].isnumeric():
                if smile[i+1].isnumeric():
                    ts=int(smile[i]+smile[i+1])
                    if int(ts) not in numbers_found:
                        if int(ts)==max(numbers_found)+1:
                            numbers_found+=[int(ts)]
                            smile.pop(i+1)
                else:
                    if int(smile[i]) not in numbers_found:
                        numbers_found+=[int(smile[i])]
                
            i+=1
        
        bondType=''#bond type of current bond
        
        current=None#last processed node
        global Elements
        if len(smile)>1:#some nodes are made from elements with 2 character names. We check for these befor single character elements (ex check for Cl before C)
            if smile[0]+smile[1] in Elements:
                current=comparisonNode(smile[0]+smile[1],ID)#create a node. Hold it as our current node
                smile=smile[2:]
            elif smile[0]=='[': #process bracketed node. Square brackets '[]' represent a set of characters which will be writen to a single node
                smile.pop(0)
                s=smile[:smile.index(']')]
                while len(s)!=1:
                    s=[s[0]+s[1]]+s[2:] #find where the square bracket ends and write contents to s
                current=comparisonNode(s[0],ID)
                smile=smile[smile.index(']')+1:]
        if not current:
            current=comparisonNode(smile.pop(0),ID)
            
        self.origin=[[current,True]]#the origin is a refrence to a node within the sructure. All other nodes can be reached from this node
        #some structures have disconnected peices. In this case multiple origins will be tracked
        
        while smile:
            char=smile.pop(0) #the next char in the smile

            t=None
                    
            if len(smile)>0:
                if char+smile[0] in Elements:
                    new=comparisonNode(char+smile.pop(0),ID)
                    bond(bondType,new,current)
                    if bondType=='.': self.origin+=[[new,True]]
                    bondType=''
                    t=new
            
            if not t:
                if char.isnumeric():
                    if char not in refs:
                        refs[char]=(current,len(self.origin)-1)
                    else:
                        if refs[char][1]!=len(self.origin)-1:
                            self.origin[-1][1]=False #if a refrence connects a disconnected structure back to another structure then mark its origin for removal
                        bond('-',current,refs[char][0]) #refrences appear to only connect through single bonds
                    
                elif char in Elements:
                    new=comparisonNode(char,ID)
                    bond(bondType,new,current)
                    if bondType=='.': self.origin+=[[new,True]]
                    bondType=''
                    t=new

                elif char=='[':
                    s=smile[:smile.index(']')]
                    while len(s)!=1:
                        s=[s[0]+s[1]]+s[2:]#find where the square bracket ends and write contents to s
                    new=comparisonNode(s[0],ID)
                    smile=smile[smile.index(']')+1:]
                    bond(bondType,new,current)
                    if bondType=='.': self.origin+=[[new,True]]
                    bondType=''
                    t=new
                
                elif char in ['-','=','.','#','$',':','/','\\']:
                    bondType=char

                elif char=='(':
                    count=1
                    i=0
                    while count!=0:
                        i+=1
                        if smile[i]=='(':
                            count+=1
                        elif smile[i]==')':
                            count-=1
                    new=smile[:i]
                    if new[0] in ['-','=','.','#','$',':','/','\\']:
                        bondType=new.pop(0)
                    smile=smile[i+1:]
                    s=structure(new,refs,ID,False)
                    #s.trace()
                    bond(bondType,s.origin[0],current)
                    bondType=''
                else:
                    print('!!!!!!!!!!!!!!!!',char,smile) #there was an error and the smile could not be decoded
            if t:
                current=t
                
        self.origin=[self.origin[0][0]]+[o[0] for o in self.origin[1:] if o[1]] #remove duplicate origins

    def trace(self): #trace origins
        for o in self.origin:
            o.trace()

    def string(self): #trace origins
        s=''
        for o in self.origin:
            s+=o.string()+'\n'
        return s

    def __str__(self):
        return self.string()

    def size(self):
        return self.origin[0].size()
        
class bond():
    '''
    A bond between two nodes
    
    Each bond connects to two nodes. these nodes can be accessed through self.links
    Each bond adds itself to its node's lists of bonds.
    Each bond has a bond type from ['-','=','.','#','$',':','/','\\']
    '''
    def __init__(self,bondType,source,dest):
        if bondType!='.':#bond type '.' mean there is no bond
            Bonds={'-':1,'=':2,'#':3,}
            ID=source._ID
            self.id=ID[1][0]
            ID[1][0]+=1
            if bondType=='':
                bondType='-' # '-' and '' are the same bond
            self.bondType=bondType
            self.links=[source,dest]

            #update local nodes
            source.add(self)
            source.sum_bonds+=Bonds[bondType]
            dest.add(self)
            dest.sum_bonds+=Bonds[bondType]

            #update global nodes
            source.all_nodes+=dest.all_nodes
            source.all_bonds+=dest.all_bonds+[self]
            _, indicesList = np.unique([n.id for n in source.all_nodes], return_index=True)
            source.all_nodes = [source.all_nodes[i] for i in indicesList]
            _, indicesList = np.unique([b.id for b in source.all_bonds], return_index=True)
            source.all_bonds = [source.all_bonds[i] for i in indicesList]
            for n in source.all_nodes:
                n.all_nodes=source.all_nodes
                n.all_bonds=source.all_bonds

    def __eq__(self,other):
        if isinstance(other,int):
            if self.id==other:
                return True
            return False
        else:
            if self.bondType==other.bondType:
                return True
            if self.bondType==':' and (other.bondType=='-' or other.bondType=='=' or other.bondType==';'):
                return True
            if other.bondType==':' and (self.bondType=='-' or self.bondType=='=' or self.bondType==';'):
                return True
            return False

    def __ne__(self,other):
        return not self.__eq__(other)

class node():
    '''
    A node is an atom in a compound (although some nodes can be multiple atoms when created via'[]'

    
    '''
    def __init__(self,value,ID):
        self.id=ID[0][0]
        self._ID=ID
        self._size=ID[0]
        ID[0][0]+=1 #ID is a global counter for all nodes within a structure
        self.all_nodes=[self]
        self.all_bonds=[]
        self.value=value
        self.miscellaneous={} # a place to store miscellaneous data Format-> {owner:data}
        if '-' == value or '-' in value[:-1] or '=' in value:
            raise(Exception(value))
        self.sum_bonds=0
        self.bonds=[] #initialize list of bonds
        self.loops=None

    def add(self,newBond):
        self.bonds+=[newBond] #add the bond to the list of bonds

    def remove(self,bond,remove_if_empty=True):
        n1,n2=bond.links
        if n2.id==self.id:
            n1,n2=n2,n1 #swap
        n1.bonds.pop([b.id for b in n1.bonds].index(bond.id))
        n2.bonds.pop([b.id for b in n2.bonds].index(bond.id))
        del bond
        if not n1.bonds:
            del n1
        if remove_if_empty and not n2.bonds:
            del n2


    def countBonds(self):
        return sum([Bonds[b.bondType] for b in self.bonds])

    def trace(self):
        print(self.string())

    def string(self):
        #Depth first search of all bonds and nodes in a compound
        visited=[self.id]
        retStr=[] if self.bonds else self.value
        #print([[l.id for l in b.links] for b in self.bonds])
        toDo=[[l for l in b.links if l.id!=self.id]+[b,(self.value,self.id)] for b in self.bonds]
        i=1
        while toDo:
            link,_bond,source=toDo.pop(0)
            #print(link.value)
            retStr+=[[_bond.id,source,_bond.bondType,(link.value,link.id)]]
            i+=1
            visited+=[link.id]
            toDo=[e for e in [[l for l in b.links if l.id not in visited]+[b,(link.value,link.id)] for b in link.bonds] if len(e)==3]+toDo
        if isinstance(retStr,str):
            return 'TRACE'+retStr
        else:
            return 'TRACE'+str(retStr)+str(len(retStr))

    def __str__(self):
        return self.string()

    def size(self):
        return self._ID[0][0]

    def num_bonds(self):
        return self._ID[1][0]

    def findloops(self,recount=False):
        if self.loops!=None and not recount:
            return self.loops
        loops=[]

        bonds=[]
        nodes=[self]
        visitedBonds=[]
        todo=[(self,b) for b in self.bonds]

        while todo:
            cn,cb=todo.pop()
            bonds+=[cb]
            visitedBonds+=[cb.id]
            if cn.id not in [n.id for n in nodes]:
                nodes+=[cn]
            todo+=[(n,b) for n in cb.links for b in n.bonds if b.id not in visitedBonds]

        starts=[(n,b,ln) for b in bonds for n,ln in (b.links,reversed(b.links))]
        
        def _findloops(cn,cb,loopedNode,visitedNodes=None):
            if visitedNodes==None:
                visitedNodes=[cn.id]
            if cn==loopedNode:
                return True,[cb.id]
            retlist=[cb.id]
            result=False
            for n,b in [(n,b) for b in cn.bonds for n in b.links if b.id!=cb.id and n.id not in visitedNodes]:
                _result,_retlist = _findloops(n,b,loopedNode,visitedNodes.copy()+[n.id])
                if _result:
                    result=_result
                    retlist+=_retlist
            return result,(retlist if result else [])

        for start_node,start_bond,looped_node in starts:
            
            loops+=_findloops(start_node,start_bond,looped_node)[1]
        self.loops=list(set(loops))
        for n in nodes:
            n.loops=self.loops
        return self.loops

    def __eq__(self,other):
        if isinstance(other,int):
            if self.id==other:
                return True
            return False
        else:
            if self.value=='?' or other.value=='?':
                return True
            if self.value==other.value:
                return True
            return False

    def __ne__(self,other):
        return not self.__eq__(other)






############### Comparison Node #######################

class comparisonNode(node):  

    def _match(self,subNode,structNode):#Match node  to node   
        if subNode!=structNode:
            return False
        Atoms={'H':1,'C':4,'O':2,'N':3,'F':1,'Cl':1,'Br':1,'I':1,'Si':4, 'P':3, 'S':2, 'O-':1, 'N-':2,'?':np.Inf,'N+':4, 'B':3}
        bondTypes={'-':1, '=':2, '#':3}
        def countBonds(n):
            return {t:sum([1 if b.bondType==t else 0 for b in n.bonds]) for t in bondTypes}
        subNodeCounts=countBonds(subNode)
        structNodeCounts=countBonds(structNode)
        for t in bondTypes:
            if subNodeCounts[t]>structNodeCounts[t]:
                return False
        return True

    
    def __match(self,possibleComp,possibleTarget,matchedNodes,matchedBonds,targetMatches,compLoops,structLoops,recursiveCall=False,pathLen=0):

        finished=False

        if  len(possibleComp)==0:
            return True,pathLen,matchedNodes,matchedBonds,finished
        if  len(possibleTarget)==0:
            return False,pathLen-1,matchedNodes,matchedBonds,finished
        
        
        i,j=0,0
        while i<len(possibleComp):
            cb,cn = possibleComp[i]
            while j<len(possibleTarget):
                tb,tn = possibleTarget[j]

                validPair=1
                for cid,tid in matchedNodes:
                    if cid==cn.id and tid==tn.id:
                        validPair=2
                        break
                    elif cid==cn.id or tid==tn.id:
                        validPair=False
                        break

                if self._match(cn,tn) and tb==cb and (((cb.id not in matchedBonds[:,0] and tb.id not in matchedBonds[:,1] and validPair==1) or  validPair==2)):
                    newMatchedNodes=np.append(matchedNodes,[[cn.id,tn.id]],axis=0)
                    if -1 in matchedBonds[:,0]:
                        newMatchedBonds=np.array([[cb.id,tb.id]])
                    else:
                        newMatchedBonds=np.append(matchedBonds,[[cb.id,tb.id]],axis=0)

                    newPossibleComp=[(b,n) for b in cn.bonds for n in b.links if b.id not in newMatchedBonds[:,0] and n.id != cn.id]
                    newPossibleTarget=[(b,n) for b in tn.bonds for n in b.links if b.id not in newMatchedBonds[:,1] and n.id != tn.id]
                    
                    
                    bResult,resultLen,newMatchedNodes,newMatchedBonds,finished=self.__match(newPossibleComp,newPossibleTarget,newMatchedNodes.copy(),newMatchedBonds.copy(),targetMatches,compLoops,structLoops,recursiveCall=True,pathLen=pathLen+1)
                    if finished:
                        return bResult,resultLen,newMatchedNodes,newMatchedBonds,finished

                    
                    if bResult:
                        matchedNodes=newMatchedNodes
                        matchedBonds=newMatchedBonds
                        pathLen=len(matchedBonds)

                        loopComp=[(cid in compLoops)==(tid in structLoops) if (cid in compLoops) else True for cid,tid in matchedBonds]
                        targLoopComp=[cid in matchedBonds[:,0] for cid in compLoops]

                        if targetMatches==pathLen and False not in loopComp and False not in targLoopComp:
                            finished=True
                            return True,pathLen,matchedNodes,matchedBonds,finished
                    
                j+=1
                
            if j>=len(possibleTarget):
                i+=1
                j=0
        
        if recursiveCall==False:
                finished=True
        return False,pathLen,matchedNodes,matchedBonds,finished


    def match(self,struct):#Match node to structure
        if self!=struct:
            return False,[]
        
        matchedParis=np.array([[self.id,struct.id]])

        possibleComp=[(sb,ll) for sb in self.bonds for ll in sb.links if ll.id != self.id]
        possibleTarget=[(sb,ll) for sb in struct.bonds for ll in sb.links if ll.id != struct.id]


        result=self.__match(possibleComp,possibleTarget,matchedParis,np.array([[-1,-1]]),self.num_bonds(),self.findloops(),struct.findloops())
        return result[0],result[-2]

    def copy(self,ID=None,copy_miscellaneous=True,preserve_ID=True):
        
        origin=self

        nID_cross_ref={}
        bID_cross_ref={}

        if ID==None:
            ID=[[0],[0]]
        retStruct=comparisonNode(origin.value,ID)
        if copy_miscellaneous:
            retStruct.miscellaneous = origin.miscellaneous.copy()
        nID_cross_ref[retStruct.id]=origin.id
        rdict={origin.id:retStruct}
        
        visited=[origin.id]
        toDo=[[l for l in b.links if l.id!=origin.id]+[b,origin] for b in origin.bonds]
        
        while toDo:
            dest,oldBond,source=toDo.pop(0)

            # rdict[link.id]=node(bondType,rdict[source.id],node(link.value,[link.id]))
            newAtom=comparisonNode(dest.value,ID)
            if copy_miscellaneous:
                newAtom.miscellaneous = dest.miscellaneous.copy()
            nID_cross_ref[newAtom.id]=dest.id
            newBond=bond(oldBond.bondType,rdict[source.id],newAtom)
            bID_cross_ref[newBond.id]=oldBond.id

            rdict[dest.id]=newAtom
                    
            visited+=[dest.id]
            toDo=[e for e in [[l for l in b.links if l.id not in visited]+[b,dest] for b in dest.bonds] if len(e)==3]+toDo

        if preserve_ID:
            for n in retStruct.all_nodes:
                n.id = nID_cross_ref[n.id]
            for b in retStruct.all_bonds:
                b.id = bID_cross_ref[b.id]
        
        return retStruct

    def copyByAtom(self):#returns max id in struct and dictionary of atoms in copied struct. Atoms represented as comparisonNodes
        
        origin=self

        ID=[[0],[0]]
        retStruct=comparisonNode(origin.value,ID)
        rdict={origin.id:retStruct}
        
        visited=[origin.id]
        toDo=[[l for l in b.links if l.id!=origin.id]+[b.bondType,origin] for b in origin.bonds]
        
        while toDo:
            dest,bondType,source=toDo.pop(0)
            
            newAtom=comparisonNode(dest.value,ID)
            bond(bondType,rdict[source.id],newAtom)

            rdict[dest.id]=newAtom
                    
            visited+=[dest.id]
            toDo=[e for e in [[l for l in b.links if l.id not in visited]+[b.bondType,dest] for b in dest.bonds] if len(e)==3]+toDo
        
        return ID[0][0], rdict

    # def _smile(self,cur,visited):
    #     retStr=''
    #     i=0
    #     for l in cur.bonds:
    #         n=[_ for _ in l.links if _.id!=cur.id][0]
    #         if n.id in visited:
    #             if n.id not in [a.id for b in cur.bonds for a in b.links]:#if b.links[0].id not in (cur.id,n.id) and b.links[1].id not in (cur.id,n.id)]:
    #                 retStr+=l.bondType+n.value+str(n.id)
    #                 i+=1
    #             else:
    #                 pass
    #             continue
    #         if i!=0:
    #             retStr+='('+str(cur.id)
    #         visited+=[n.id]
    #         retStr+=l.bondType+str(n.id)+n.value+self._smile(n,visited)
    #         if i!=0:
    #             retStr+=')'
    #         i+=1
        
    #     return retStr
    
    # def smile(self):
    #     if len(self.bonds)>0:
    #         return self._smile(self,[])[1:]
    #     else:
    #         return self.value

    def _smile(self,cur,visited):
        retStr=''
        for i,l in enumerate(cur.bonds):
            n=[_ for _ in l.links if _.id!=cur.id][0]
            if n.id in visited:
                if n.id not in [a.id for b in cur.bonds for a in b.links]:#if b.links[0].id not in (cur.id,n.id) and b.links[1].id not in (cur.id,n.id)]:
                    retStr += l.bondType + n.value
                    i+=1
                else:
                    pass
                continue
            if i<len(cur.bonds):
                retStr+='('
            visited+=[n.id]
            retStr += l.bondType + n.value + self._smile(n,visited)
            if i<len(cur.bonds):
                retStr+=')'
        
        return retStr
    
    def smile(self):
        if len(self.bonds)>0:
            return self._smile(self,[])[1:]
        else:
            return self.value