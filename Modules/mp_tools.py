from tqdm.auto import tqdm
import sys
import multiprocessing as mp
mp.set_start_method('spawn', True)
import marshal
import types
import inspect

import time

def _work(DataQ, ReturnQ, fun, constants, serial=False):
    if not serial:
        
        try:
            args,k = DataQ.get()
            args+=constants
            ReturnQ.put((k, fun(*args)))
            return True
        except Exception as e:
            print('\nERROR!:')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("\n"+str(exc_tb.tb_lineno)+" "+str(exc_obj)+" "+str(exc_tb),"\n")
            ReturnQ.put((k,e))
            return True
    else:
        args,k = DataQ.get()
        args+=constants
        #print(args)
        ReturnQ.put((k,fun(*args)))
        return True

def _worker(JobsDoneQ, NofJobs, NofWorkers, ReturnQ, DataQ, fun, constants):
    # Worker loop
    working = True
    if not inspect.isclass(fun):
        fun = types.FunctionType(marshal.loads(fun), globals(), "fun")
    while working:
        jobNo = JobsDoneQ.get()
        _work(DataQ, ReturnQ, fun, constants)
        if NofJobs-jobNo <= NofWorkers-1:
            working = False

class Buddy():

    def __new__(cls, fun, jobsD, constants=None, serial=False,cores=None,verbose=False):
        resultsQ=mp.Queue()

        constants=[] if constants==None else [constants] if not isinstance(constants,list) else constants
        
        if (not inspect.isclass(fun)) and not serial:
            fun = marshal.dumps(fun.__code__)
        
        JobLen=len(jobsD)
        if cores==None:
            CORES = max(1,min(mp.cpu_count(),JobLen))
        else:
            CORES=cores
        if verbose:
            print('Using',CORES,'cores')

        JobsDoneQ=mp.Queue()
        ReturnQ=mp.Queue()
        ReadRequestQ=mp.Queue()
        DataQ=mp.Queue()
        DataBuffer=min(CORES,JobLen)
        keys=list(jobsD.keys())
        
        for i in range(JobLen):
            JobsDoneQ.put(i+1)
            ReadRequestQ.put(1)
        
        for i in range(DataBuffer):
            k=keys[i]
            DataQ.put((jobsD[k],k))
            ReadRequestQ.get()
            ReadRequestQ.put(0)
        
        

        if not serial:
            p = {}
            for core in range(CORES):
                p[core] = mp.Process(target=_worker,
                                  args=[JobsDoneQ, JobLen, CORES, ReturnQ, DataQ, fun, constants])
                p[core].start()

        results={}
        
        #Read returned data from workers, add new read reqest
        r=range(DataBuffer, JobLen+DataBuffer)
        for i in (r if not verbose else tqdm(r)):
            if serial:
                _work(DataQ, ReturnQ, fun, constants, serial=True)
                if serial=='p': print(i, DataBuffer, JobLen+DataBuffer)
            r=ReturnQ.get()
            results[r[0]]=r[1]
            #print(iCompound,cType)
            if ReadRequestQ.get():
                k=keys[i]
                DataQ.put((jobsD[k], k))

        if not serial:
            for core in range(CORES):
                p[core].terminate()
                p[core].join()

        return results

