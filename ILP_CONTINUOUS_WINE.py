from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import accuracy_score ,precision_recall_curve,auc,precision_recall_fscore_support
import operator
import scipy.signal


# for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4
TEST_SET_INDEX=0

#load data and create 5 datasets for 5-fold test

DATA_FILE = './data/wine.data'
data = np.genfromtxt(DATA_FILE, delimiter=',')

data_x = data[:,1:]
data_y = data[:,0]-1

data_xm = np.mean( data_x,axis=0,keepdims=True)
data_xv = np.std( data_x,axis=0,keepdims=True)
data_x = (data_x-data_xm) / data_xv 

np.random.seed(0)

L = data_x.shape[0] // 5 
inds = np.random.permutation( L*5)
DataSets=[]

for i in range(5):
    DataSets.append(  (data_x[inds[ i*L:(i+1)*L],:] , data_y[inds[ i*L:(i+1)*L]]))

names=['Alc','MA','Ash', 'AlcAsh','Mag','Phn','Flav','Nonf', 'Pro', 'Col','Hue' , '0D','Proline' ]
 	
#define predicates 

Constants = dict({})
predColl = PredCollection (Constants)
 
for i in range(13): 
    predColl.add_continous(name=names[i],no_lt=4,no_gt=4)
    
for i in range(3): 
    predColl.add_pred(name='class_%d'%(i+1),arguments=[] , variables=[] , pFunc = DNF('class_%d'%(i+1),terms=2,init=[-1,.1,-1,.1],sig=2)   ,use_cnt_vars=True,inc_preds=[])

predColl.initialize_predicates()    

#add backgrounds

bg_train=[]
bg_test=[]

for j in range(5):
    for i in range(L):
        
        bg = Background( predColl ) 
        
        
        bg.add_example(pred_name='class_1',pair=( ), value= float(DataSets[j][1][i]==0) )
        bg.add_example(pred_name='class_2',pair=( ), value= float(DataSets[j][1][i]==1) )
        bg.add_example(pred_name='class_3',pair=( ), value= float(DataSets[j][1][i]==2) )

        for k in range(13):
            bg.add_continous_value( names[k], ( (DataSets[j][0][i,k] ,),) )
            
        if j == TEST_SET_INDEX:    
            bg_test.append(bg)
        else:
            bg_train.append(bg)
 
BS = len(bg_test)
def bgs(it,is_train):

    if is_train:
        n=it%4
        return bg_train[ n*L:(n+1)*L]
        # inds= np.random.permutation(L*4)
        return [ bg_train[inds[i]] for i in range(L) ]
    else:
        return bg_test
    
    
  
# ###########################################################################

def disp_fn(eng,it,session,cost,outp):
    
    Y_true=[]
    Y_score=[]
    
    cl1 = outp['class_1']
    cl2 = outp['class_2']
    cl3 = outp['class_3']
    
    for i in range(L):
        Y_true.append(    DataSets[TEST_SET_INDEX][1][i])
        Y_score.append( np.argmax( [cl1[i][0] ,cl2[i][0],cl3[i][0] ]) )

            
    acc = accuracy_score(Y_true, Y_score)
    print('***********************************')
    print('accuracy score = ',  acc)   
    print('***********************************')
 
    return

     
parser = argparse.ArgumentParser()

parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=1,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=BS,help='Batch Size',type=int)
parser.add_argument('--T',default=1,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.0051} , help='Learning rate schedule',type=dict)
parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
parser.add_argument('--MAXTERMS',default=10 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=.001 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.001 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
parser.add_argument('--FILT_TH_MEAN', default=.2 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=.2 , help='Fast convergence total loss threshold MAX',type=float)
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

parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)

args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    


