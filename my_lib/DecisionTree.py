import numpy as np
import pandas as pd

class Decision_Tree_Classifier:

    def __init__(self):
        self.tree=None

    def entropy(self,y):
        if len(y)!=0:
            p=sum(y)/len(y)
        else:
            p=0
        if p==0 or p==1:
            return 0
        else:
            return -p*np.log2(p) - (1-p)*np.log2(1-p)
        
    def split (self,X,index,feature):
        left=[]
        right=[]
        for i,x in zip(index,X[index]):
            if x[feature]==1:
                left.append(i)
            else:
                right.append(i)
        return left,right
    
    def weighted_entropy(self,X,y,index,left,right):
        w1=len(left)/(len(X[index]))
        w2=len(right)/(len(X[index]))
        w_entropy=w1*self.entropy(y[left])+w2*self.entropy(y[right])
        return w_entropy
    
    def info_gain(self,X,y,index,feature,left_index,right_index):
        node_entropy=self.entropy(y[index])
        split_entropy=self.weighted_entropy(X,y,index,left_index,right_index)
        gain=node_entropy-split_entropy
        return gain

    def best_split(self,X,y,index):
        feature_count=X.shape[1]
        best_gain=-1
        for feature in range(feature_count):
            l_index,r_index=self.split(X,index,feature)
            gain=self.info_gain(X,y,index,feature,l_index,r_index)
            if gain>best_gain:
                best_gain=gain
                best_l_index=l_index
                best_r_index=r_index
                best_feature=feature
        return best_l_index,best_r_index,best_gain,best_feature

    def build_tree(self,X,y,index=None,max_depth=999,current_depth=0):
        if current_depth==0:
            index=[i for i in range(len(y))]
        if self.entropy(y[index])==0:
            return y[index][0]
        l_index,r_index,gain,feature=self.best_split(X,y,index)
        if (current_depth==max_depth) or (gain==0):
            return y[index][0] 
        # print(l_index,r_index)
        self.tree={
            feature: {
                'left': self.build_tree(X,y,l_index,max_depth,current_depth + 1),
                'right': self.build_tree(X,y,r_index,max_depth,current_depth + 1)
            }
        }
        return self.tree