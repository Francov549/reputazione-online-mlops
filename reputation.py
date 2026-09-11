"""Pipeline didattica MLOps: inferenza, valutazione e retraining controllato."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
DATASET_ID = 'cardiffnlp/tweet_eval'
LABELS = ['negative', 'neutral', 'positive']
SEED = 42


def normalize(text):
    """Preserva negazioni, emoji e maiuscole; uniforma menzioni e URL."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Il testo deve essere una stringa non vuota')
    return ' '.join(' '.join('http' if w.startswith('http') else '@user' if w.startswith('@') and len(w)>1 else w for w in text.split()).split())


def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError('Etichette mancanti o lunghezze diverse')
    return {'n':len(y), 'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'negative_recall':float(recall_score(y,pred,labels=[0],average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}


def promotion(base, candidate):
    """Soglie definite a priori: miglioramento F1 e tutela della classe negativa."""
    return bool(candidate['macro_f1'] >= 0.60 and
                candidate['macro_f1'] >= base['macro_f1'] + 0.005 and
                candidate['negative_recall'] >= base['negative_recall'] - 0.02)


def js_divergence(p, q):
    p, q = np.asarray(p,dtype=float)+1e-8, np.asarray(q,dtype=float)+1e-8
    p, q = p/p.sum(), q/q.sum()
    m=(p+q)/2
    return float((np.sum(p*np.log2(p/m))+np.sum(q*np.log2(q/m)))/2)


def monitor(reference, probabilities, labels=None, reference_f1=None):
    """Drift delle predizioni != degrado della qualita; richiede etichette umane."""
    probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.ndim!=2 or probabilities.shape[1]!=3 or len(probabilities)==0:
        raise ValueError('Servono probabilita non vuote con forma (n,3)')
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,atol=1e-5):
        raise ValueError('Probabilita non valide')
    pred=probabilities.argmax(1)
    shares=np.bincount(pred,minlength=3)/len(pred)
    drift=js_divergence(reference,shares)
    report={'n':len(pred),'shares':shares.tolist(),'negative_share':float(shares[0]),
            'mean_confidence':float(probabilities.max(1).mean()),
            'low_confidence_share':float((probabilities.max(1)<0.60).mean()),
            'js_divergence':drift,'reputation_alert':bool(len(pred)>=30 and shares[0]>=0.50),
            'prediction_drift_alert':bool(len(pred)>=30 and drift>0.10),
            'performance_alert':False, 'needs_retraining':False}
    if labels is not None:
        report['quality']=metrics(labels,pred)
        report['performance_alert']=bool(len(pred)>=30 and reference_f1 is not None and report['quality']['macro_f1']<reference_f1-0.05)
        report['needs_retraining']=report['performance_alert']
    return report


class SentimentModel:
    def __init__(self, model_id=MODEL_ID, revision=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from huggingface_hub import HfApi
        self.torch=torch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        pins=json.loads(Path('revisions.json').read_text()) if Path('revisions.json').exists() else {}
        revision=revision or (pins.get('model') if model_id==MODEL_ID else None)
        self.revision=revision or (None if Path(model_id).exists() else HfApi().model_info(model_id).sha)
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=self.revision)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=self.revision).to(self.device)
        actual=[self.model.config.id2label[i].lower() for i in range(3)]
        if actual!=LABELS:
            raise ValueError(f'Ordine etichette inatteso: {actual}')
        self.model.eval()

    def predict(self, texts, batch_size=32):
        clean=[normalize(t) for t in texts]
        if not clean:
            return np.empty((0,3)), {'seconds':0.0,'texts_per_second':0.0}
        self.model.eval()
        out=[]
        started=time.perf_counter()
        with self.torch.inference_mode():
            for i in range(0,len(clean),batch_size):
                tokens=self.tokenizer(clean[i:i+batch_size],padding=True,truncation=True,max_length=128,return_tensors='pt').to(self.device)
                out.append(self.model(**tokens).logits.softmax(-1).cpu().numpy())
        elapsed=time.perf_counter()-started
        return np.concatenate(out), {'seconds':elapsed,'texts_per_second':len(clean)/elapsed}

    def save(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path,safe_serialization=True)
        self.tokenizer.save_pretrained(path)

    def train_head(self, texts, labels, epochs=2, learning_rate=1e-4):
        """Congela RoBERTa; aggiorna la testa classificatrice per un retraining leggero."""
        torch=self.torch
        torch.manual_seed(SEED)
        random.seed(SEED)
        for param in self.model.roberta.parameters():
            param.requires_grad=False
        optimizer=torch.optim.AdamW(self.model.classifier.parameters(),lr=learning_rate,weight_decay=0.01)
        logs=[]
        for epoch in range(epochs):
            self.model.train()
            self.model.roberta.eval()
            order=np.random.default_rng(SEED+epoch).permutation(len(texts))
            total=0.0
            for offset in range(0,len(order),16):
                ids=order[offset:offset+16]
                batch=self.tokenizer([normalize(texts[int(i)]) for i in ids],padding=True,truncation=True,max_length=128,return_tensors='pt').to(self.device)
                y=torch.tensor([int(labels[int(i)]) for i in ids],device=self.device)
                optimizer.zero_grad()
                loss=self.model(**batch,labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(),1.0)
                optimizer.step()
                total+=float(loss.detach())*len(ids)
            logs.append({'epoch':epoch+1,'loss':total/len(texts)})
        self.model.eval()
        return logs


def dataset_splits(n_train=1200,n_val=600,n_test=1200,revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    pins=json.loads(Path('revisions.json').read_text()) if Path('revisions.json').exists() else {}
    revision=revision or pins.get('dataset') or HfApi().dataset_info(DATASET_ID).sha
    raw=load_dataset(DATASET_ID,'sentiment',revision=revision)
    pools={}
    seen=set()
    audit={}
    # Test protetto per primo; rimuove duplicati e sovrapposizioni normalizzate.
    for split in ['test','validation','train']:
        rows=[]
        for row in raw[split]:
            try:
                key=digest(row['text'])
            except ValueError:
                continue
            if key not in seen and row['label'] in [0,1,2]:
                rows.append({'text':row['text'],'label':int(row['label']),'id':key})
                seen.add(key)
        pools[split]=rows
        audit[split]={'original':len(raw[split]),'valid_unique':len(rows)}
    selected={}
    for split,n in [('train',n_train),('validation',n_val),('test',n_test)]:
        rows=pools[split]
        if n and n<len(rows):
            ids,_=train_test_split(np.arange(len(rows)),train_size=n,stratify=[r['label'] for r in rows],random_state=SEED)
            rows=[rows[int(i)] for i in ids]
        selected[split]=rows
        audit[split]['sampled']=len(rows)
        audit[split]['class_counts']=np.bincount([r['label'] for r in rows],minlength=3).tolist()
    return selected, {'dataset':DATASET_ID,'revision':revision,'audit':audit}, pools


def run_experiment(output='artifacts',n_train=1200,n_val=600,n_test=1200,feedback=None):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    splits,metadata,pools=dataset_splits(n_train,n_val,n_test)
    train=splits['train']
    feedback_count=0
    if feedback:
        # Non usa pseudo-label; le etichette devono essere state validate da persone.
        protected={r['id'] for k in ['validation','test'] for r in pools[k]}
        train_ids={r['id'] for r in train}
        incoming=Path('data/incoming.jsonl')
        if incoming.exists():
            protected.update(digest(json.loads(line)['text']) for line in incoming.read_text().splitlines() if line.strip())
        for line in Path(feedback).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('human_verified') is not True or type(row.get('label')) is not int or row['label'] not in [0,1,2]:
                raise ValueError('Feedback richiede label 0/1/2 e human_verified=true')
            key=digest(row['text'])
            if key not in protected and key not in train_ids:
                train.append({'text':row['text'],'label':row['label'],'id':key})
                train_ids.add(key)
                feedback_count+=1
        if feedback_count<30:
            raise ValueError('Servono almeno 30 nuovi esempi umani unici e non sovrapposti')
    print('Caricamento modello richiesto...',flush=True)
    engine=SentimentModel(os.getenv('CHAMPION_MODEL_ID') or MODEL_ID)
    base_state={k:v.detach().cpu().clone() for k,v in engine.model.classifier.state_dict().items()}
    val=splits['validation']
    val_text=[r['text'] for r in val]
    val_y=[r['label'] for r in val]
    base_prob,base_latency=engine.predict(val_text)
    base_metrics=metrics(val_y,base_prob.argmax(1))
    print('Validazione baseline:',base_metrics,flush=True)
    logs=engine.train_head([r['text'] for r in train],[r['label'] for r in train])
    candidate_prob,_=engine.predict(val_text)
    candidate_metrics=metrics(val_y,candidate_prob.argmax(1))
    accepted=promotion(base_metrics,candidate_metrics)
    print('Validazione candidato:',candidate_metrics,'| promosso:',accepted,flush=True)
    engine.save(out/'candidate')
    if not accepted:
        engine.model.classifier.load_state_dict(base_state)
    # Test finale solo sul modello scelto sulla validation.
    test=splits['test']
    test_prob,latency=engine.predict([r['text'] for r in test])
    test_y=[r['label'] for r in test]
    majority=int(np.argmax(np.bincount([r['label'] for r in train],minlength=3)))
    final_metrics=metrics(test_y,test_prob.argmax(1))
    rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(300):
        ids=rng.integers(0,len(test_y),len(test_y))
        boot.append(f1_score(np.array(test_y)[ids],test_prob.argmax(1)[ids],labels=[0,1,2],average='macro',zero_division=0))
    engine.save(out/'champion')
    reference_prob=candidate_prob if accepted else base_prob
    reference=(np.bincount(reference_prob.argmax(1),minlength=3)/len(val)).tolist()
    report={'seed':SEED,'model':MODEL_ID,'model_revision':engine.revision,'data':metadata,
            'training':{'method':'frozen_roberta_train_classifier','epochs':2,'learning_rate':1e-4,'n':len(train),'new_human_feedback':feedback_count,'logs':logs},
            'validation_base':base_metrics,'validation_candidate':candidate_metrics,'candidate_promoted':accepted,
            'selected_model':'candidate' if accepted else 'baseline','test':final_metrics,
            'test_macro_f1_bootstrap95':np.quantile(boot,[0.025,0.975]).tolist(),
            'majority_test':metrics(test_y,[majority]*len(test_y)),
            'latency':latency,'device':engine.device,'reference_shares':reference,
            'reference_f1':(candidate_metrics if accepted else base_metrics)['macro_f1']}
    save_json(out/'report.json',report)
    save_json(out/'test_predictions.json',{'ids':[r['id'] for r in test],'labels':test_y,'probabilities':test_prob.tolist()})
    import importlib.metadata as im
    save_json(out/'environment.json',{p:im.version(p) for p in ['torch','transformers','datasets','numpy','scikit-learn','huggingface-hub']})
    print('Esperimento completato:',out/'report.json',flush=True)
    return engine,report,splits


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='artifacts')
    p.add_argument('--n-train',type=int,default=1200)
    p.add_argument('--n-val',type=int,default=600)
    p.add_argument('--n-test',type=int,default=1200)
    p.add_argument('--feedback')
    args=p.parse_args()
    run_experiment(**vars(args))
