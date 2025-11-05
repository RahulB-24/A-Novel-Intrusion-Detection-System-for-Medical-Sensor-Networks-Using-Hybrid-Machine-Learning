import numpy as np, pandas as pd, os, json
from scipy.signal import sawtooth

OUT = os.path.dirname(__file__)

def generate_sensor_sample(seq_len=128, hr_mean=70, temp_mean=36.6, spo2_mean=97):
    t = np.linspace(0, 1, seq_len)
    hr   = hr_mean + 5*np.sin(2*np.pi*1.5*t) + np.random.normal(0,1,seq_len)
    temp = temp_mean + 0.2*np.sin(2*np.pi*0.2*t) + np.random.normal(0,0.05,seq_len)
    spo2 = spo2_mean + 0.5*np.sin(2*np.pi*0.5*t) + np.random.normal(0,0.2,seq_len)
    ecg  = 0.5*sawtooth(2*np.pi*5*t,0.5) + 0.05*np.random.randn(seq_len)
    return np.stack([hr,temp,spo2,ecg],axis=1)

def inject_attack(sample, kind):
    s = sample.copy()
    if kind=="spoof": s[:,0]*=np.random.uniform(1.2,1.6)
    elif kind=="replay": s[20:40]=s[0:20]
    elif kind=="injection": s[50:55,3]+=np.random.uniform(1.0,2.5,5)
    return s

def make_dataset(n_normal=1500,n_attack=500,seq_len=128):
    rng=np.random.default_rng(42); X=[]; y=[]
    for _ in range(n_normal):
        X.append(generate_sensor_sample(seq_len))
        y.append(0)
    kinds=["spoof","replay","injection"]
    for _ in range(n_attack):
        s=generate_sensor_sample(seq_len)
        s=inject_attack(s,rng.choice(kinds))
        X.append(s); y.append(1)
    X=np.stack(X).astype(np.float32); y=np.array(y)
    np.save(os.path.join(OUT,"processed_timeseries.npy"),X)
    np.save(os.path.join(OUT,"processed_flattened.npy"),X.reshape(len(X),-1))
    np.save(os.path.join(OUT,"labels.npy"),y)
    json.dump({"n_normal":n_normal,"n_attack":n_attack},open(os.path.join(OUT,"metadata.json"),"w"))
    print("Dataset generated:",X.shape,y.shape)
    return X,y

if __name__=="__main__": make_dataset()
