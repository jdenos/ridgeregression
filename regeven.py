import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
from PyQt4 import QtGui, QtCore
from ui import Ui_MainWindow



global Ntr, Nte, d, Xte, Xtr, ytr, yte, Wopt, bopt, modified, trainf, testf, predf
modified = False
testf = ''
trainf = ''
predf = ''
Ntr = 0
Nte = 0
d = 0
Xte = []
yte = []
Xtr = []
ytr = []


def lireData():
    global Ntr, Nte, d, Xte, Xtr, ytr, yte, modified, trainf, testf
    d=0
    if (testf != "" and trainf !=""):
        test = open(testf,'r')
        train = open(trainf,'r')
        litest=test.readlines()
        litrain=train.readlines()

        Ntr = len(litrain)
        Nte = len(litest)
        # lit les lignes et compte d

        for l in litest:
            a =l.split(" ")
            yte+= [float(a[0])]
            l=[]
            for j in range(1,len(a)):
                t = a[j].split(":")
                t[0]=int(t[0])
                t[1]=int(t[1])
                if t[0] > d :
                    d= t[0]
                l+= [t]
            Xte+=[l]

        for l in litrain:
            a =l.split(" ")
            ytr+= [float(a[0])]
            l=[]
            for j in range(1,len(a)):
                t = a[j].split(":")
                t[0]=int(t[0])
                t[1]=int(t[1])
                if t[0] > d :
                    d= t[0]
                l+= [t]
            Xtr+=[l]
        #transformation en matrice
        yte = np.array(yte)
        ytr = np.array(ytr)
        MXtr = np.zeros((Ntr,d),dtype=np.int)
        MXte = np.zeros((Nte,d),dtype=np.int)

        i=0
        for l in Xtr :
            for couple in l :
                MXtr[i,couple[0]-1]=couple[1]
            i+=1
        i=0
        for l in Xte :
            for couple in l :
                MXte[i,couple[0]-1]=couple[1]
            i+=1

        Xtr = MXtr
        Xte = MXte
        modified = False
        return 0
    elif (trainf == "") :
        print "entrer un fichier train SVP"
    else :
        print "entrer un fichier test SVP"
    modified = False
    return 1

#calcul W pour un b donner
def calW(b):
    t1 =np.dot(Xtr.T,Xtr)+ b*np.identity(d,dtype=np.int)
    t1=LA.inv(t1)
    t2=np.dot(Xtr.T,ytr)
    W=np.dot(t1,t2)
    return W
def altcalW(b) :
    t1 =np.dot(Xtr.T,Xtr)+ b*np.identity(d,dtype=np.int)
    ch = LA.cholesky(t1)
    ch=LA.inv(ch)
    t1=np.dot(ch.T,ch)
    t2=np.dot(Xtr.T,ytr)
    W=np.dot(t1,t2)
    return W
#sors les stats sur le set extern
def statsext(W):
    ytepred = np.dot(Xte,W)
    SSE = (LA.norm(ytepred-yte,2))**2
    RMSE = (1/float(Nte))*np.sqrt(SSE)
    ytemoy=np.mean(yte)
    SST = (LA.norm(yte-ytemoy))**2
    R2=1-SSE/SST
    return RMSE,R2
#sors les stats sur le set d'entrainement
def statintern(W):
    ytrpred = np.dot(Xtr,W)
    SSE = (LA.norm(ytrpred-ytr,2))**2
    RMSE = (1/float(Ntr))*np.sqrt(SSE)
    ytrmoy=np.mean(ytr)
    SST = (LA.norm(ytr-ytrmoy))**2
    R2=1-SSE/SST
    return RMSE,R2
#trace en fonction de b par increment
def plotb(mini,maxi,pas):
    i=mini
    bvect =[]
    RMSEvect=[]
    R2vect=[]
    while (i<=maxi):
        RMSE,R2 = statsext(calW(i))
        bvect+=[i]
        RMSEvect+=[RMSE]
        R2vect +=[R2]
        i+=pas
    fig=plt.figure()
    subf=fig.add_subplot(111)
    p1=subf.plot(bvect,RMSEvect,'bs-')
    subf.set_ylabel("RMSE")
    plt.xlabel("b")
    subf2=subf.twinx()
    p2 = subf2.plot(bvect,R2vect,'rs-')
    subf2.set_ylabel("R2")
    lns=p1+p2
    labs=["RMSE","R2"]
    subf.legend(lns, labs)
    plt.show()
#RMSE en fonction de b
def statRMSE(b):
    RMSE = statsext(calW(b))
    return RMSE

#optimise b via la recherche de minimum par dichotomie
def optimiseb(a,b,c,conv):
    if c-a<=conv :
        print c-a
        return b
    if (c-b>=b-a) :
        d=c-b
    else :
        d=a-b
    x = b+0.39197*d
    if (x<b):
        if statRMSE(x) < statRMSE(b) :
            return optimiseb(a,x,b,conv)
        else :
            return optimiseb(x,b,c,conv)
    else:
        if statRMSE(x)< statRMSE(b):
            return optimiseb(b, x, c, conv)
        else :
            return optimiseb(a,b,x,conv)

#trace RMSE,R2 = f(b) en logaritme @param puidi:puissance de 10 maximum souhaite
def plotlog(puidi):
    bvect =[]
    RMSEvect=[]
    R2vect=[]
    for i in range(0,puidi):
        for j in range(1,10):
            RMSE,R2=statsext(calW(j*10**i))
            bvect+=[j*10**i]
            RMSEvect+=[RMSE]
            R2vect+=[R2]

    fig=plt.figure()
    subf=fig.add_subplot(111)
    p1=subf.plot(bvect,RMSEvect,'bs-')
    subf.set_ylabel("RMSE")
    plt.xlabel("b")
    subf2=subf.twinx()
    p2 = subf2.plot(bvect,R2vect,'rs-')
    subf2.set_ylabel("R2")
    lns=p1+p2
    labs=["RMSE","R2"]
    subf.legend(lns, labs)
    plt.show()

def plotResidu(W) :
    ypred = np.dot(Xte,W)
    fig=plt.figure()
    subf=fig.add_subplot(111)
    p1=subf.plot(yte,ypred,'bs')
    subf.set_ylabel("Ypred")
    plt.xlabel("Y")
    plt.show()

def selecttrain():
    global trainf , modified
    s= QtGui.QFileDialog.getOpenFileName()
    trainf = s
    modified = True
    if (trainf != ""):
        window.TestBtn.setEnabled(True)
        window.MsgTxt.append("Fichier d'entrainement charge")
    else :
        window.MsgTxt.append("aucun fichier charge")

def selecttest():
    global testf, modified
    s = QtGui.QFileDialog.getOpenFileName()
    testf = s
    modified = True
    if (testf != "") :
        window.TraceBox.setEnabled(True)
        window.LogTraceBox.setEnabled(True)
        window.modeleBox.setEnabled(True)
        window.MsgTxt.append("Fichier test charge")
    else :
        window.MsgTxt.append("aucun fichier charge")


def plotbAction() :
    if (modified) :
        lireData()
    try :
        a = float(window.TraceMinTxt.text())
        b= float(window.TraceMaxTxt.text())
        c = float(window.TracePasTxt.text())
    except ValueError :
        window.MsgTxt.append("erreur d'arguments")
    if (a<b):
        plotb(a,b,c)
    else :
        plotb(b,a,c)


def plotlogAction() :
    if (modified):
        lireData()
    n = int(window.LTDixPuisTxt.text())
    plotlog(n)
def modeleAction() :
    global bopt, Wopt
    if (modified):
        lireData()
    if (window.PredefBRBtn.isChecked()) :
        bopt = float(window.BTxt.text())
    else :
        a = float(window.BMinTxt.text())
        c = float(window.BMaxTxt.text())
        b= a+(c-a)/2
        pas= float(window.IntervalTxt.text())
        bopt = optimiseb(a,b,c,pas)
    Wopt=calW(bopt)
    window.ResBtn.setEnabled(True)
    window.statsBtn.setEnabled(True)
    window.PredBox.setEnabled(True)

def ResiduAction() :
    global Wopt
    plotResidu(Wopt)
def statsAction() :
    global Wopt,bopt

    RMSE,R2 = statsext(Wopt)
    affi = "pour b = "+str(bopt)+" la RMSE est de "+str(RMSE)+" et R2 = "+str(R2)
    window.MsgTxt.append(affi)
def selectPredAction():
    global predf
    predf=''
    s= QtGui.QFileDialog.getOpenFileName()
    if (s != ''):
        predf = s
    if (predf != '') :
        window.PredBtn.setEnabled(True)
        window.MsgTxt.append("Fichier de molecules a predire charge")
    else :
        window.MsgTxt.append("aucun fichier charge")

def Predict(W) :
    global predf,d

    pred = open(predf,"r")
    lipred = pred.readlines()

    Npred = len(lipred)
    mat=[]
    for l in lipred :
        a =l.split(" ")
        l=[]
        for j in range(len(a)):
            t = a[j].split(":")
            t[0]=int(t[0])
            t[1]=int(t[1])
            l+= [t]
        mat +=[l]
    Xpred = np.zeros((Npred,d),dtype=np.int)
    i=0
    for l in mat :
        for couple in l :
            Xpred[i,couple[0]-1]=couple[1]
        i+=1

    return np.dot(Xpred,W)
def predictAction():
    global Wopt
    ypred = Predict(Wopt)
    for i in range(len(ypred)) :
        window.MsgTxt.append("molecule" + str(i+1)+" : "+str(ypred[i]))


class Ridge(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        window=Ui_MainWindow.__init__(self)
        # Configure l'interface utilisateur.
        self.setupUi(self)
        self.connect(self.TrainBtn,QtCore.SIGNAL("clicked()"),selecttrain)
        self.connect(self.TestBtn,QtCore.SIGNAL("clicked()"),selecttest)
        self.connect(self.ModelBtn,QtCore.SIGNAL("clicked()"),modeleAction)
        self.connect(self.ResBtn,QtCore.SIGNAL("clicked()"),ResiduAction)
        self.connect(self.TraceBtn,QtCore.SIGNAL("clicked()"),plotbAction)
        self.connect(self.LTBtn,QtCore.SIGNAL("clicked()"),plotlogAction)
        self.connect(self.statsBtn,QtCore.SIGNAL("clicked()"),statsAction)
        self.connect(self.SelectPredBtn,QtCore.SIGNAL("clicked()"),selectPredAction)
        self.connect(self.PredBtn,QtCore.SIGNAL("clicked()"),predictAction)

#main
app = QtGui.QApplication(sys.argv)
window = Ridge()
window.show()
sys.exit(app.exec_())

'''testf='test.svm'
trainf='train.svm'
lireData()'''