from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import roc_auc_score ,precision_recall_curve,auc,precision_recall_fscore_support,accuracy_score,confusion_matrix
import operator
import scipy.signal


# use numbers 1 to 5 to evaluate each of 5 datafiles
SILICO_NUM =5




TIME_SERIES_FILE = './data/silico%d.txt'%SILICO_NUM
GOLD_STANDARD_FILE = './data/gold%d.txt'%SILICO_NUM
WT_FILE = './data/wt%d.tsv'%SILICO_NUM



def my_roc_auc_score(x,y):
    return np.round(1000*roc_auc_score(x,y))/1000.
def get_score(y_true,y_pred,th):
    # print(confusion_matrix(y_true, [ float(i>=th) for i in y_pred])).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, [ float(i>=th) for i in y_pred]).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall=tp/(tp+fn)
    fscore=2*precision*recall/(precision+recall)
    MCC=(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*tn+fn))
    return np.round(1000*acc)/1000.,np.round(1000*fscore)/1000.,np.round(1000*MCC)/1000.


# read gold standard
true_class=OrderedDict()
with open(GOLD_STANDARD_FILE) as f:
    for s,d,r in read_by_tokens(f):
        i = int(s[1:])
        j = int(d[1:])
        r = float(r)
        true_class[ (i,j)] = r
         

        

# read data files
data = np.loadtxt(TIME_SERIES_FILE)
data = data[:,1:]


for i in range(10):
    pass
    data[:,i] = scipy.signal.medfilt(data[:,i]  ,3)
    data[:,i] = np.convolve(data[:,i], np.ones((11,))/11, mode='same')

wt = np.loadtxt(WT_FILE)
wt = np.expand_dims( wt, 0)
s = np.std(data,axis=0,keepdims=True)
data=data-wt
data=data/s


fivept = np.percentile(np.abs(data),15,axis=0)
nightfivept = np.percentile(np.abs(data),100-15,axis=0)




Constants = dict({})
 


predColl = PredCollection (Constants)
def mygmm(name='',num=2):
    x_m = bias_variable( (num,),np.linspace(-1.,1.,num,dtype=np.float32),name='mean_'+name)
    x_m = tf.clip_by_value( x_m, -4.5,4.5)
        
    x_s = bias_variable( (num,),value=np.linspace(.5,1.5,num,dtype=np.float32),name='std_'+name)
    x_s = tf.clip_by_value( x_s, .1,8.)
        
    x_c = bias_variable( (num,),value=np.linspace(.1,.9,num,dtype=np.float32),name='cp_'+name)
    x_c = tf.nn.softmax(x_c)

        
    cat = tf.distributions.Categorical( logits=x_c)  
    dists = [tf.distributions.Normal( x_m[i],x_s[i]) for i in range(num)]
    pxm = tf.contrib.distributions.Mixture( cat=cat,components=dists)
    return pxm

# predicate function class for predicate 'Off' 
class MyFuncOffGMMD(PredFunc):
    def __init__(self,name='',trainable=True,predColl=None,index=1):
        
        super().__init__(name,trainable)
        self.index = index
        self.predColl = predColl
        
        self.x_m=None
        self.x_s=None
        self.x_c=None
     
    def pred_func(self,xi,xcs,t):
        d1 = mygmm('d1',6)
        d2 = mygmm('d2',6)
        pn = 'G_%d'%(self.index)
        x = xcs[pn][:,:,t]     
        p1 = d1.prob(x)
        p2 = d2.prob(x)

        ws = bias_variable( (2,),value=[.5,.5],name='c'+self.name)
        ws = tf.nn.softmax( ws)

        a1 =  p1*ws[0] 
        a2 =  p2*ws[1] 
        y = a1/(a1+a2+1e-5)

        return y
    
    def get_func(self,session,names,threshold=.2,print_th=True):
        return ""
        m,s,c = session.run( [self.x_m,self.x_s,self.x_c])
        c=np.exp(c)/np.sum(np.exp(c))
        return '---'.join( [ 'Normal(%.2f,%.2f,%.2f)'%(a1,a2,a3)   for  a1,a2,a3 in zip(m,s,c) ] )


# predicate function class for predicate 'aux' 
class MyFuncAux(PredFunc):
    def __init__(self,name='',trainable=True,predColl=None,index=1):
        
        super().__init__(name,trainable)
        self.index = index
        self.predColl = predColl
    def pred_func(self,xi,xcs,t):
        
        onc_name = 'off_%d'%(self.index)
        on_name = 'inf_off%d'%(self.index)
        target_name = 'aux_%d'%(self.index)
        ind1 = self.predColl.preds_by_name[target_name].inp_list.index(on_name+'()')
        ind2 = self.predColl.preds_by_name[target_name].inp_list.index(onc_name+'()')
        
        
        a=xi[:,ind1] 
        b=xi[:,ind2] 
        
        res = 1 - tf.abs(a-b)
        return res
    
    def get_func(self,session,names,threshold=.2,print_th=True):
        return ""  
 
#define all predicates
for i in range(10): 
 
    predColl.add_continous(name='G_%d'%(i+1),no_lt=0,no_gt=0)
    predColl.add_pred(name='off_%d'%(i+1),arguments=[] , variables=[] , pFunc = MyFuncOffGMMD(name='off_%d'%(i+1), predColl=predColl,index=i+1) ,use_cnt_vars=True,inc_preds=[] ,inc_cnt=['G_%d'%(i+1)] )
    
    from_pred = [ 'off_%d'%(j+1) for j in range(10) if j!=i ]
    predColl.add_pred(name='inf_off%d'%(i+1),arguments=[] , variables=[] , pFunc =CNF('inf_off%d'%(i+1),terms=1,init=[6,.1,-1,.1],sig=2)  ,use_cnt_vars=False,inc_preds=from_pred,use_neg=False )
    predColl.add_pred(name='aux_%d'%(i+1),arguments=[] , variables=[] , pFunc = MyFuncAux(predColl=predColl,index=i+1)  ,use_cnt_vars=False,inc_preds=None )

predColl.initialize_predicates()    
bgss=[]
    
# add background facts
for j in range(1):
    for i in range(1,105):
        bg = Background( predColl ) 
        
        
        x = data[i]
         
        for k in range(10):
            bg.add_example(pred_name='aux_%d'%(k+1),pair=( ) , value=1. )
            
            if abs(x[k])>=nightfivept[k]:
                bg.add_example(pred_name='off_%d'%(k+1),pair=( ) , value=0.  )
            if abs(x[k])<=fivept[k]:
                bg.add_example(pred_name='off_%d'%(k+1),pair=( ) , value=1.)

            bg.add_continous_value( 'G_%d'%(k+1), ( (x[k],),) )
        bgss.append(bg)

 
BS = len(bgss)
def bgs(it,is_training):
    return bgss
  
# ###########################################################################

def disp_fn(eng,it,session,cost,outp):
    links={}
    
    for i in range(10):
        

        preds =  predColl['inf_off%d'%(i+1)]
        target =  predColl['aux_%d'%(i+1)]
        items=preds.pFunc.get_item_contribution(session,preds.inp_list , threshold=.001)
        factors = items


        for j in range(10):
            predd =  'off_%d()'%(j+1)
            if predd not in items :
                continue
            lk = (i+1,j+1)
            fac = 0.0
            if predd in factors:
                fac = factors[predd]

             
            update_dic( links, lk, fac)
            
            
 
         
            
    links_list = sorted(links.items(), key=operator.itemgetter(1))
    links_list=links_list[::-1]
     
    res_items=OrderedDict()
    for lk,ranking in links_list:
        pi='off_%d'%(lk[0])
        pj='off_%d'%(lk[1])
        oi = outp[pi][:,0]
        oj = outp[pj][:,0]

        nj = np.sum( (1.0-oi)*(1.0-oj)) / (1.0e-3+ np.sum( 1.0-oi) )
        ni = np.sum( (1.0-oj)*(1.0-oi)) / (1.0e-3+ np.sum( 1.0-oj) )
        
        if ni<=nj :
            update_dic(res_items,(lk), ranking )
        else:
            update_dic(res_items,(lk[::-1]), ranking )
    
    if res_items:
        maxv = np.max( list(res_items.values() ))+1e-3
    else:
        maxv=1
    Y_true=[]
    Y_score=[]
    for item in true_class:
        Y_true.append(true_class[item])
        
        if item in  res_items:
            vv = res_items[item]/maxv
            Y_score.append( min( 1.0, vv) )
        else:
            Y_score.append(0.)
    Y_score=np.array(Y_score)
     
    print('***************************************') 
    
    for t in [ .9]:
        print(get_score(Y_true,Y_score,t))   
 
parser = argparse.ArgumentParser()


parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=BS,help='Batch Size',type=int)
parser.add_argument('--T',default=1,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.0051} , help='Learning rate schedule',type=dict)


parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)

parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--USE_OR',default=1 ,help='Use Or in updating value vectors',type=int)
parser.add_argument('--SIG',default=1,help='sigmoid coefficient',type=int)
parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
parser.add_argument('--N1',default=1,help='softmax N1',type=int)
parser.add_argument('--N2',default=1,help='Softmax N2',type=int)
parser.add_argument('--FILT_TH_MEAN', default=-.2 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=-.2 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--ITER', default=100*20, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=100, help='Epoch',type=int)
parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)
parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    


