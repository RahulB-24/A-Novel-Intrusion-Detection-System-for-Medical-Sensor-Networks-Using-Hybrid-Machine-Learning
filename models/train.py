import numpy as np, os
from models.hybrid_model import build_hybrid
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

ROOT=os.path.dirname(os.path.dirname(__file__))
DATA=os.path.join(ROOT,"data")
CKPT=os.path.join(ROOT,"models","checkpoints"); os.makedirs(CKPT,exist_ok=True)

def train():
    X=np.load(os.path.join(DATA,"processed_timeseries.npy"))
    y=np.load(os.path.join(DATA,"labels.npy"))
    from sklearn.model_selection import train_test_split
    Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    m=build_hybrid(Xtr.shape[1],Xtr.shape[2])
    cb=[ModelCheckpoint(os.path.join(CKPT,"hybrid_best.h5"),save_best_only=True,monitor='val_accuracy',mode='max'),
        EarlyStopping(patience=6,restore_best_weights=True),
        ReduceLROnPlateau(patience=3)]
    m.fit(Xtr,Ytr,validation_data=(Xte,Yte),epochs=25,batch_size=32,callbacks=cb)
    m.save(os.path.join(CKPT,"hybrid_final.h5"))
    print("Training complete.")

if __name__=="__main__": train()
