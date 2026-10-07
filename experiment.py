"""Reproducible experimental experiment; all transforms fit training data only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
def classification(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p).tolist(),'report':classification_report(y,p,output_dict=True,zero_division=0)}
def neural(x,y,vx,vy,tx,regression=False):
    import torch,copy
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(42)
    x,vx,tx=[torch.tensor(z,dtype=torch.float32) for z in (x,vx,tx)]
    scale=max(float(np.std(y)),1.) if regression else 1.
    mean=float(np.mean(y)) if regression else 0.
    y,vy=[torch.tensor((z-mean)/scale,dtype=torch.float32).reshape(-1,1) if regression else torch.tensor(z,dtype=torch.long) for z in (y,vy)]
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1 if regression else 2))
    criterion=nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    best=float('inf');weights=None;wait=0;history=[]
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=criterion(model(x[batch]),y[batch]);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode(): score=float(criterion(model(vx),vy))
        history.append(score)
        if score<best-1e-5:best=score;weights=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=12:break
    model.load_state_dict(weights)
    with torch.inference_mode():out=model(tx)
    pred=out.numpy().ravel()*scale+mean if regression else out.argmax(1).numpy()
    return pred,{'epochs':len(history),'best_validation_loss':best,'target_scale':scale,'target_mean':mean}
def experiment(data=None):
    from sklearn.datasets import load_iris
    data=load_iris();x,y=data.data,data.target
    a,b,ya,yb=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    models={'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3)),'DecisionTree':DecisionTreeClassifier(max_depth=3,random_state=42),'RandomForest':RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)}
    results={}
    for name,model in models.items():model.fit(a,ya);results[name]=classification(yb,model.predict(b))
    return {'dataset':'sklearn Iris','train':len(a),'test':len(b),'seed':42,'models':results}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path,default=Path('metrics.json'))
    args=parser.parse_args()
    results=experiment(args.data)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results))
