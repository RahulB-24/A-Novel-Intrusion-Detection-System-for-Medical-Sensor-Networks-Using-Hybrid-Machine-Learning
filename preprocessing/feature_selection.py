import numpy as np, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from collections import Counter

ROOT=os.path.dirname(os.path.dirname(__file__))
DATA=os.path.join(ROOT,"data")

def select_features(k=12):
    X=np.load(os.path.join(DATA,"train.npy"))
    y=np.load(os.path.join(DATA,"train_labels.npy"))
    rf=RandomForestClassifier(n_estimators=100,random_state=42).fit(X,y)
    rf_idx=np.argsort(rf.feature_importances_)[-k:]
    chi=SelectKBest(chi2,k=k).fit(X-X.min()+1e-6,y)
    mi =SelectKBest(mutual_info_classif,k=k).fit(X,y)
    vote=Counter(list(rf_idx)+list(chi.get_support(indices=True))+list(mi.get_support(indices=True)))
    sel=[i for i,c in vote.items() if c>=2][:k]
    np.save(os.path.join(ROOT,"preprocessing","selected_idx.npy"),sel)
    print("Selected indices:",sel)
    return sel

if __name__=="__main__": select_features()
