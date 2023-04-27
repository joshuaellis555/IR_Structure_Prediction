"""
Program: IRSpectrum.py
Programmed by: Josh Ellis, Josh Hollingsworth, Aaron Kruger, Alex Matthews, and
    Joseph Sneddon
Description: This program will recieve an IR Spectrograph of an unknown
    molecule and use our algorithm to compare that graph to a stored database of
    known molecules and their IR Spectrographs. This program will then return a
    list of the closest Spectrographs matches as determined by our algorithm.
IR_Functions.py: This part of the program contains most of the functions used by
    Query.py and UpdatedDB.py.
"""
#---------------------------------Imports--------------------------------------
##import PyPDF2
import sqlite3
from PIL import Image
import sys
import warnings
import os

def LoadPickle(file,verbose=False,encoding='ASCII'):     
    import pickle

    if verbose: print(file)
    
    f=open(file,'rb')
    spec=pickle.load(f,encoding=encoding)
    f.close()
    return spec

def SavePickle(data,path):
    import pickle
            
    f=open(path,'wb+')
    pickle.dump(data,f)
    f.close()

def MakeSpectrum(fname,spectra,bold=True):
    #print(fname)
    from PIL import Image
    import sys
    import os
    from shutil import copyfile
    
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    ensure_dir("\\".join(fname.split("\\")[:-1]))
    
    width = -113+978+1
    height = -29+724+1

    xMin=200
    xMax=4100
    xRange=xMax-xMin #the x-range of the graph
    yMin=1.02
    yMax=-0.05
    yRange=yMax-yMin #the y-range of the graph

    def deconvertx(x):
        
        try:
            #return xMin+xRange*(x/1024)
            return min(width-1,max(0,round((x-xMin)*(width)/xRange)))
        except:
            print(x,width,xRange)
            raise

    def deconverty(y):
        
        #return yMin+yRange*(y/768)
        return min(height-1,max(0,round((y-yMin)*(height)/yRange)))

    def convertx(x):
        return xMin+xRange*(x/width)
    def converty(y):
        return yMin+yRange*(y/height)

    graph=[(255,255,255) if sum(p)>600 else (0,0,0) for p in list(Image.open('graphTemplate.png').getdata())]

    def avgpix(x,y,c):
        r,g,b=graph[y*(1024)+x]
        graph[y*(1024)+x]=int((r+c[0])/2),int((g+c[1])/2),int((b+c[2])/2)
        
    for spectrum in spectra:

        if len(spectrum)==2:
            color1,data1=spectrum
            dataset=[data1]
            colorset=[color1]
        elif len(spectrum)==5:
            color1,data1,color2,data2,fillcolor=spectrum
            dataset=[data1,data2]
            colorset=[color1,color2]

        for k in range(len(dataset)+1):
            if k!=0:
                data=dataset[k-1]
                color=colorset[k-1]
            else:
                data=dataset[0]
                
            preY=None
            i=0
            while i<len(data):
                tx=data[i][0]
                    
                x=int(deconvertx(tx)+113)
                if data[i][1]==None:
                    preY=None
                    continue
                ty=deconverty(data[i][1])+29

                if len(dataset)==2 and k==0:
                    ty2=deconverty(dataset[1][i][1])+29

                    for y in range(min(ty,ty2),max(ty,ty2)+1):
                        graph[int(y*(1024)+x)]=fillcolor

                if k!=0:
                    if preY:
                        for y in range(min(ty,preY),max(ty,preY)+1):
                            graph[y*(1024)+x]=color
                            if bold:
                                avgpix(x-1,y,color)
                                avgpix(x+1,y,color)
                                avgpix(x,y-1,color)
                                avgpix(x,y+1,color)
                    else:
                        graph[int(ty*(1024)+x)]=color
                    preY=ty

                i+=1
            
    
    r = Image.new('RGB',(1024,768))
    r.putdata(graph)
    r.save(fname)
#------------------------------------------------------------------------------
