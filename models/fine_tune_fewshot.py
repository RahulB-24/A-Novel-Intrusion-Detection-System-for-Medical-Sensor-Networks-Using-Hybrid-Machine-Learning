import numpy as np, os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

ROOT=os.path.dirname(os.path.dirname(__file__))
DATA=os.path.join(ROOT,"data")
CKPT=os.path.join(ROOT,"models","checkpoints")

def fine_tune(few_shot=20,epochs=5):
    model=load_model(os.path.join(CKPT,"hybrid_final.h5"))
    for layer in model.layers[:-2]: layer.trainable=False
    model.compile(optimizer=Adam(1e-4),loss='binary_crossentropy',metrics=['accuracy'])
    X=np.load(os.path.join(DATA,"processed_timeseries.npy"))
    y=np.load(os.path.join(DATA,"labels.npy"))
    # simulate few-shot adaptation on few new samples
    idx=np.random.choice(len(X),few_shot,replace=False)
    Xs,ys=X[idx],y[idx]
    model.fit(Xs,ys,epochs=epochs,batch_size=8,verbose=1)
    model.save(os.path.join(CKPT,"hybrid_finetuned.h5"))
    print("Few-shot fine-tuning done.")

if __name__=="__main__": fine_tune()
