import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
import tensorflow as tf
import os.path
from .mylibw import *
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import *


class ILPRLEngine(object):

    def __init__(self,args,predColl,bgs,disp_fn=None ):
        print( 'Tensorflow Version : ', tf.__version__)
        
        self.args=args
        self.predColl = predColl 
        self.predColl.args=args
        self.bgs=bgs
        self.disp_fn=disp_fn
        tf.set_random_seed(self.args.SEED) 
        config=tf.ConfigProto( device_count = {'GPU': self.args.GPU} )
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session= tf.Session(config=config)

        self.plogent =  tf.placeholder("float32", [],name='plogent') 
        self.index_ins=OrderedDict({})
        self.X0 = OrderedDict({})
        self.target_mask = OrderedDict({})
        self.target_data = OrderedDict({})
        for p in self.predColl.preds:
           
            self.index_ins[p.name] = tf.constant( self.predColl.InputIndices[p.name])
            self.X0[p.name] =          tf.placeholder("float32", [self.args.BS,self.predColl[p.name].pairs_len] , name='input_x_' + p.name)
            if p.pFunc is None:
               continue
            self.target_data[p.name] = tf.placeholder("float32", [self.args.BS,self.predColl[p.name].pairs_len] , name='target_data_' + p.name)
            self.target_mask[p.name] = tf.placeholder("float32", [self.args.BS,self.predColl[p.name].pairs_len] , name='target_mask_' + p.name)

        self.thresholds_lt={}
        self.thresholds_gt={}
        self.continuous_inputs={}
        for v in self.predColl.cnts:
            self.thresholds_gt[v.name] = tf.get_variable( 'th_gt_' + v.name , shape=(v.no_gt), initializer=tf.constant_initializer(v.gt_init, dtype=tf.float32) ,collections=[ tf.GraphKeys.GLOBAL_VARIABLES,'CONTINUOUS'])
            self.thresholds_lt[v.name] = tf.get_variable( 'th_lt_' + v.name , shape=(v.no_lt), initializer=tf.constant_initializer(v.lt_init, dtype=tf.float32) ,collections=[ tf.GraphKeys.GLOBAL_VARIABLES,'CONTINUOUS'] )
            self.continuous_inputs[v.name] = tf.placeholder("float32", [self.args.BS,v.dim,self.args.T] ,name='continous_input_'+v.name)
        
        
        self.define_model()
        print("summary all variables")
        
        for k in tf.trainable_variables():
            if( isinstance(k, tf.Variable) and len(k.get_shape().as_list())>1 ):
                print( str(k))
                if( self.args.TB==1):
                    tf.summary.histogram( k.name, k)
                    if len(k.get_shape().as_list())==2:
                        tf.summary.image(k.name, tf.expand_dims( tf.expand_dims(k,axis=0),axis=3))
                    if len(k.get_shape().as_list())==3:
                        tf.summary.image(k.name, tf.expand_dims( k,axis=3))

        if self.args.TB==1:
            self.all_summaries = tf.summary.merge_all()
    ############################################################################################
    def check_weights( self,sess , w_filt  ):
        
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue
                wv_sig = p.pFunc.conv_weight_np(wv)
                
                sumneg = np.sum( np.logical_and(wv_sig>.1,wv_sig<.9))
                if sumneg > 0: 
                    print( "weights in %s are not converged yet :  %f"%(wt.name,sumneg))
                    return False
        return True
    ############################################################################################
    def filter_predicates( self,sess , w_filt,th=.5  ):
        
        old_cost,_=self.runTSteps(sess)
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue
                
                
                wv_sig = p.pFunc.conv_weight_np(wv)
                for ind,val in  np.ndenumerate(wv_sig):
                    
                    if val>.5:
                        wv_backup = wv*1.0
                        wv[ind]=-20
                        sess.run( wt.assign(wv))
                        cost,_=self.runTSteps(sess)
                         
                        if cost-old_cost >th :
                            wv = wv_backup*1.0
                        else:
                            old_cost=cost
                            print( 'removing',wt,ind)
                            
                sess.run( wt.assign(wv))
    ############################################################################################
    def filter_predicates2( self,sess ,  th=.5  ):
        
        old_cost,_=self.runTSteps(sess)
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )
            
            wand = None
            wor = None
            if 'AND' in wts[0].name:
                wand = wvs[0]
                wor = wvs[1]
                wandt = wts[0]
                wort = wts[1] 
            else:
                wand = wvs[1]
                wor = wvs[0]
                wandt = wts[1]
                wort = wts[0]
            
            wand_bk=wand*1.0
            wor_bk=wand*1.0

            wand_sig = p.pFunc.conv_weight_np(wand)
            wor_sig = p.pFunc.conv_weight_np(wor)

            
             
            for k in range(wor_sig[0,:].size):
                if wor_sig[0,k]>.1:
                    
                    wor[0,k]=-20
                    sess.run( wort.assign(wor))
                    cost,_=self.runTSteps(sess)
                    
                    if abs(cost-old_cost) >.1 :
                        wor[0,k] = wor_bk[0,k]
                    else:
                        old_cost=cost
                        print( 'removing',wort,k)
                        continue

                    for v in range( wand_sig[k,:].size):
                        if wand_sig[k,v]>.1:
                            
                            wand[k,v]=-20
                            sess.run( wandt.assign(wand))
                            cost,_=self.runTSteps(sess)
                            
                            if abs(cost-old_cost) >.1 :
                                wand[k,v] = wand_bk[k,v]
                            else:
                                old_cost=cost
                                print( 'removing',wandt,v)
                                continue


            sess.run( wort.assign(wor))
            sess.run( wandt.assign(wand))
    ############################################################################################
    def get_sensitivity_factor( self,sess , p ,target_pred ):
        
        target_data = self.SARG[self.target_data[target_pred.name]]
        target_mask  = self.SARG[self.target_mask[target_pred.name]]

        def getval():
            val =  sess.run( self.XOs[target_pred.name]  , self.SARG )
            err = np.sum(  (val-target_data)*target_mask )
            return err

        
        
        # old_cost,_=self.runTSteps(sess)
        old_cost = getval()


        factors = dict({})
        wts = tf.get_collection( p.name)
        if len(wts)==0:
            return factors
        wvs = sess.run( wts )

        
        for wt,wv in zip(wts,wvs):
             
            if 'AND' not in wt.name:
                continue
            
            wv_sig = p.pFunc.conv_weight_np(wv)
            wv_backup = wv*1.0

            wv = wv_backup*1.0
            wv[:]=-20 
            sess.run( wt.assign(wv))
            cost_all = getval()
            cost_all_diff = abs( cost_all-old_cost)+1e-3

            
            # print('val',val)
            for k in  range(wv_sig[0,:].size):
                
                if np.max( wv_sig[:,k] ) <.1:
                    continue
                wv = wv_backup*1.0
                
                wv[:,k]=-20 
                sess.run( wt.assign(wv))
                # cost,_=self.runTSteps(sess)
                cost=getval()    
                if abs(cost-old_cost) >1 :
                    sens=1.0
                else:
                    sens = (1e-3+abs(cost-old_cost) )/cost_all_diff

                if k<=len(p.inp_list):
                    item = p.inp_list[ k]
                    factors[item]=sens

            return factors
    ############################################################################################
    def get_sensitivity_factor1( self,sess , p ,target_pred ):
        
        target_data = self.SARG[self.target_data[target_pred.name]]
        target_mask  = self.SARG[self.target_mask[target_pred.name]]

        def getval():
            val =  sess.run( self.XOs[target_pred.name]  , self.SARG )
            err = np.sum(  (val-target_data)*target_mask )
            return err

        
        
        # old_cost,_=self.runTSteps(sess)
        old_cost = getval()


        factors = dict({})
        wts = tf.get_collection( p.name)
        if len(wts)==0:
            return factors
        wvs = sess.run( wts )

        
        for wt,wv in zip(wts,wvs):
             
            if 'AND' not in wt.name:
                continue
            
            wv_sig = p.pFunc.conv_weight_np(wv)
            wv_backup = wv*1.0

            wv = wv_backup*1.0
            wv[:]=-20
            sess.run( wt.assign(wv))
            cost_all = getval()
            cost_all_diff = abs( cost_all-old_cost)+1e-3

            
            # print('val',val)
            for ind,val in  np.ndenumerate(wv_sig):
                
                if val<.1:
                    continue
                wv = wv_backup*1.0
                
                wv[ind]=-20
                sess.run( wt.assign(wv))
                # cost,_=self.runTSteps(sess)
                cost=getval()    
                if abs(cost-old_cost) >1 :
                    sens=1.0
                else:
                    sens = (1e-3+abs(cost-old_cost) )/cost_all_diff

                if ind[-1]<=len(p.inp_list):
                    item = p.inp_list[ ind[-1]]
                    if item in factors:
                        factors[item] = max( factors[item], sens)
                    else:
                        factors[item]=sens

            return factors
    ############################################################################################
    def binarize( self,sess   ):
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )
            for wt,wv in zip(wts,wvs):
                wv = wv*1.6
                s = 20
                wv [ wv>s] =s
                wv[wv<-s] = -s
                sess.run( wt.assign(wv))
    ############################################################################################
    def define_model(self):
        
       
        XOs = OrderedDict( self.X0 )
        L3=0
        self.XOTS=OrderedDict()
        
        
        for t in range(self.args.T):

            olditem=OrderedDict()
            for i in XOs:
                olditem[i] = tf.identity( XOs[i])
        
            for p in self.predColl.preds:
                
                if p.name=="CNT":
                    lenp = len(p.pairs)
                    px = np.zeros( (self.args.BS,lenp) , np.float)
                    if t<lenp:
                        px[:,t]=1
                    else:
                        px[:,-1]=1
                    
                    XOs[p.name] = tf.constant(px,tf.float32)
                    continue

                if p.pFunc is None:
                    continue


                if len( self.predColl.cnts) >0 : 
                    inp_continous=[]
                    for v in self.predColl.cnts:
                        if v.dim>1:
                            continue
                        x1 = self.continuous_inputs[v.name][:,:,t] - tf.expand_dims( self.thresholds_gt[v.name] , 0)
                        x2 = self.continuous_inputs[v.name][:,:,t] - tf.expand_dims( self.thresholds_lt[v.name] , 0)
                        x1 = sharp_sigmoid( x1 , 20)
                        x2 = sharp_sigmoid( -x2 , 20)
                        
                        cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
                        cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt

                        if cond1 or cond2:
                            pass
                            # inp_continous.append(x1*0)
                            # inp_continous.append(x2*0)
                        else:
                            inp_continous.append(x1)
                            inp_continous.append(x2)

                    if len(inp_continous)==0:
                        len_continous = 0 
                    else:

                        inp_continous = tf.concat(inp_continous,-1)
                        len_continous = inp_continous.shape.as_list()[-1]
                else:
                    len_continous = 0

                if self.args.SYNC==1:
                    x = tf.concat( list( olditem.values() ), -1) 
                else:
                    x = tf.concat( list( XOs.values() ), -1) 
                
                xi=tf.gather( x  ,self.index_ins[p.name],axis=1  )
                s = xi.shape.as_list()[1]*xi.shape.as_list()[2]
                    
                    

                    
                self.xi = xi
                if p.Lx>0:
                    xi = tf.reshape( xi, [-1,p.Lx])
                    if p.use_neg:
                        xi = tf.concat( (xi,1.0-xi) ,-1)
                if len( self.predColl.cnts) >0  and p.use_cnt_vars:
                    cnt_s = tf.tile( inp_continous,(s,1) )
                    if p.Lx>0:
                        xi = tf.concat(  (xi,cnt_s),-1)
                    else:
                        xi = cnt_s
               
                
                l = xi.shape.as_list()[0]
                # if t==0:
                #     print( 'input size for F (%s) = %d'%(p.name,l))
               
                
                
                with tf.variable_scope( "ILP", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                    xi = p.pFunc.pred_func(xi,self.continuous_inputs,t)
                    if type( xi ) in  (tuple,list):
                        for a in xi[1]:
                            self.continuous_inputs[a] = copy.deepcopy( xi[1][a] )
                        xi = xi[0]

                
                

                xi = tf.reshape( xi  , [self.args.BS,]+self.index_ins[p.name].shape.as_list()[:2] )
                xi =  1.0-and_op( 1.0-xi,-1) 
                
                L3+=  tf.reduce_max( xi*(1.0-xi))
                if p.Fam =='and':
                    XOs[p.name] = XOs[p.name] *  tf.identity(xi) 
                if p.Fam =='eq':
                    XOs[p.name] = tf.identity(xi)
                if p.Fam =='or':
                    XOs[p.name] = 1.0 - (1.0-XOs[p.name] )*  (1.0-tf.identity(xi) )
                

            self.XOTS[t] = dict()
            self.XOTS[t].update(XOs)
                
                
        L1=0
        L2=0
        
        for p in  self.predColl.preds:
            vs = tf.get_collection( p.name)
            for wi in vs:
                if '_AND' in wi.name:
                    wi = p.pFunc.conv_weight(wi)
                    
                    L2 += tf.reduce_mean( wi*(1.0-wi))

                    s = tf.reduce_sum( wi,-1)
                    L1 += tf.reduce_mean(  tf.nn.relu( s-self.args.MAXTERMS)  )
        

        self.XOs=XOs
        self.loss_gr = tf.constant(0.,tf.float32)
        self.loss = tf.constant(0.,tf.float32)

        for p in self.predColl.preds:
            if p.pFunc is None:
                continue

            if self.args.L2LOSS==1:
                err = ( self.target_data[p.name] - XOs[p.name] ) * self.target_mask[p.name]
                err = tf.square(err)
                self.loss_gr +=   tf.reduce_mean(err,-1)  
            else:
                err = neg_ent_loss (self.target_data[p.name] , XOs[p.name] ) * self.target_mask[p.name]
                self.loss_gr +=  tf.reduce_mean(err,-1)  
            
           
        
            loss =  neg_ent_loss (self.target_data[p.name] , XOs[p.name] ) * self.target_mask[p.name]
            self.loss += tf.reduce_sum ( loss )  
        
        
        self.loss_gr  += ( self.args.L1*L1 + self.args.L2*L2+self.args.L3*L3 )
        
        
        self.lastlog=10
        self.cnt=0
        self.counter=0
        self.SARG=None
    ############################################################################################
    # execute t step forward chain 
    def runTSteps(self,session,is_train=False,it=-1):
         
        
        # if self.SARG is None:
        self.SARG = dict({})
        bgs = self.bgs(it,is_train)
        self.SARG[self.plogent] =self.args.PLOGENT
    
        for p in self.predColl.preds:
    
            self.SARG[self.X0[p.name]] = np.stack( [bg.get_X0(p.name) for bg in bgs] , 0 )
            if p.pFunc is None:
                continue
            self.SARG[self.target_data[p.name]] = np.stack( [ bg.get_target_data(p.name) for bg in bgs] , 0 )
            self.SARG[self.target_mask[p.name]] = np.stack( [ bg.get_target_mask(p.name) for bg in bgs] , 0 )
        
        
        for c in self.predColl.cnts:
            self.SARG[self.continuous_inputs[c.name]] =   np.stack( [ bg.continuous_vals[c.name] for bg in bgs] , 0 ) 

          
        self.SARG[self.LR] = .001
        try:
            if is_train:
                if bool(self.args.LR_SC) :
                    for l,r in self.args.LR_SC:
                        if self.lastlog >= l and self.lastlog < r:
                            self.SARG[self.LR] = self.args.LR_SC[(l,r)]
                            break

        except:
            self.SARG[self.LR] = .001
        
        
         
        
        
        
        if is_train:
            _,cost,outp =  session.run( [self.train_op,self.loss,self.XOs ] , self.SARG)
        else:
            cost,outp,xots =  session.run( [self.loss,self.XOs ,self.XOTS ] , self.SARG ) 
            
        try:
            self.lastlog = cost
        except:
            pass
        return cost,outp
    ############################################################################################
    def train_model(self):
        
        session  = self.session
        
        t1 =  datetime.now()
        print ('building optimizer...')
        self.LR = tf.placeholder("float", shape=(),name='learningRate')
        
        
        loss = tf.reduce_mean(self.loss_gr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, beta1=self.args.BETA1, 
                       beta2=self.args.BETA2, epsilon=self.args.EPS, 
                       use_locking=False, name='Adam')
         
        self.train_op  = self.optimizer.minimize(loss)
        
       
        
        t2=  datetime.now()
        print ('building optimizer finished. elapsed:' ,str(t2-t1))

        init = tf.global_variables_initializer()
        session.run(init)


        if self.args.TB==1:
            train_writer = tf.summary.FileWriter(self.args.LOGDIR, session.graph)
            train_writer.close()

        print( '***********************')
        print( 'number of trainable parameters : {}'.format(count_number_trainable_params()))
        print( '***********************')

        
        start_time = datetime.now()
         
 
        for i in range(self.args.ITER):
            

            cost,outp= self.runTSteps(session,True,i)
            
            
            if i % self.args.ITER2 == 0 and not np.isnan(np.mean(cost)):
                
            
                cost,outp= self.runTSteps(session,False,i)
                
                if self.disp_fn is not None:
                    self.disp_fn(self, i//self.args.ITER2,session,cost,outp)
                
                now = datetime.now()
                print('------------------------------------------------------------------')
                errs=OrderedDict({})
                for p in self.predColl.preds:
                    if p.pFunc is None:
                        continue
                    if np.sum(self.SARG[self.target_mask[p.name]]) >0:
                        errs[p.name] = np.sum (  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]] )
        
                print( 'epoch=' ,i//self.args.ITER2 , 'cost=', np.mean(cost),  'elapsed : ',  str(now-start_time)   ,'mismatch counts', errs)
                names=[]
                
                #displaying outputs ( value vectors)
                for bs in self.args.DISP_BATCH_VALUES:
                    if bs>0:
                        break
                    cnt = 0
                    
                    for p in self.predColl.preds:
                        print_names=[]
                        if p.pFunc is None:
                            continue
                        

                        mask = self.SARG[self.target_mask[p.name]]
                        target = self.SARG[self.target_data[p.name]]
                        if np.sum(mask) >0:
                            
                            for ii in range( p.pairs_len ):
                                if mask[bs,ii]==1:
                                    if cnt<self.args.MAX_DISP_ITEMS:
                                        print_names.append(  '[('+ ','.join( p.pairs[ii]) +')],[%2.01f,%d]  '%(outp[p.name][ bs,ii],target[bs,ii]))
                                        if  abs(outp[p.name][bs,ii]-target[bs,ii]) >.3:
                                            print_names[-1] = '*'+print_names[-1]
                                        else:
                                            print_names[-1] = ' '+print_names[-1]
                                            
                                            
                                        if  cnt%10==0:
                                            print_names[-1] = '\n' +print_names[-1]
                                        cnt+=1
                                    
                                    else:
                                        break
                        print( ' , '.join(print_names) )
                 

               

                # remove unncessary terms if near optimzed solution is achieved or preprogrammed to do so
                err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None]
                errmax = np.max (  [ e.max() for e in err])
                try:
                    
                    if i>0 and ( (i//self.args.ITER2)%self.args.ITEM_REMOVE_ITER==0) :
                        print ( 'start removing non necessary clauses')
                        self.filter_predicates(session,'OR:0')  
                        self.filter_predicates(session,'AND:0')  
                except:
                    pass
                if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax<self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):
                    
                    should_remove=True
                    for ii in range(20):
                        cost,outp= self.runTSteps(session,False)
                        err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None]
                        errmax = np.max (  [ e.max() for e in err])

                
                        if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax <self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):
                            pass
                        else:
                            should_remove = False
                            break
                    should_remove = should_remove
                    if should_remove:
                        print ( 'start removing non necessary clauses')

                        self.filter_predicates(session,'OR:0')  
                        self.filter_predicates(session,'AND:0')  
                        if self.args.BINARAIZE==1:
                            self.binarize(session)
                            
                            self.filter_predicates(session,'OR')  
                            self.filter_predicates(session,'AND')  
                            cost,outp= self.runTSteps(session,False)
                        
                        if self.args.CHECK_CONVERGENCE==1:
                            self.check_weights(session,'AND:0') 
                            self.check_weights(session,'OR:0') 

                
                # display learned predicates
                if self.args.PRINTPRED :
                    try:
                        gt,lt = session.run( [self.thresholds_gt,self.thresholds_lt] )
                        
                        for p in self.predColl.preds:

                            cnt_names = self.predColl.get_continous_var_names(p,gt,lt)
        
                            if p.pFunc is None:
                                continue
                            if p.use_cnt_vars:
                                inp_list = p.inp_list  + cnt_names
                            else:
                                inp_list = p.inp_list  

                            if p.pFunc is not None:
                                s = p.pFunc.get_func( session,inp_list,threshold=self.args.W_DISP_TH)
                                if s is not None:
                                    if len(s)>0: 
                                        print( p.name+ '(' + ','.join(variable_list[0:p.arity]) + ')  \n'+s)

                    except:
                        print('there was an exception in print pred')
                # display raw membership weights for predicates
                if self.args.PRINT_WEIGHTS==1:
                    
                    wts = tf.trainable_variables( )
                    wvs = session.run( wts )
                    for t,w in zip( wts,wvs):
                        if '_SM' in t.name:
                            print( t.name, np.squeeze( w.argmax(-1) ) )
                        else:
                            print( t.name, myC( p.pFunc.conv_weight_np(w) ,2) )
            
                
                
                # check for optimization
                err = [  (np.abs(outp[p.name]-self.SARG[self.target_data[p.name]] )) * self.SARG[self.target_mask[p.name]]  for p in self.predColl.preds if p.pFunc is not None ]
                errmax = np.max (  [ e.max() for e in err])
                
                if np.mean(cost)<self.args.OPT_TH  and ( np.mean(cost)<.0 or  errmax<.09 ):
                    

                    if self.args.CHECK_CONVERGENCE==1:
                        if self.check_weights(session,'OR:0')  and self.check_weights(session,'AND:0')  :
                        
                            print('optimization finished !')
                            
                            return
                    else:
                        print('optimization finished !')
                            
                        return

                start_time=now