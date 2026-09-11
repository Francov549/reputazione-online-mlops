"""Controllo un lotto di dati prima di decidere se avviare il retraining."""
import json
import os
from pathlib import Path
from reputation import SentimentModel, MODEL_ID, monitor, save_json

def main():
    incoming=Path('data/incoming.jsonl')
    reference_file=Path('reports/reference.json')
    if not incoming.exists():
        status={'status':'no_new_data','needs_retraining':False}
    elif not reference_file.exists():
        raise RuntimeError('Manca reports/reference.json: creare prima una baseline')
    else:
        rows=[json.loads(line) for line in incoming.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            status={'status':'empty_batch','needs_retraining':False}
        else:
            engine=SentimentModel(os.getenv('CHAMPION_MODEL_ID',MODEL_ID))
            p,timing=engine.predict([r['text'] for r in rows])
            # Valuto le prestazioni solo se tutte le etichette sono verificate.
            verified=all(r.get('human_verified') is True and type(r.get('label')) is int and r['label'] in [0,1,2] for r in rows)
            y=[r['label'] for r in rows] if verified else None
            reference=json.loads(reference_file.read_text())
            status=monitor(reference['reference_shares'],p,y,reference['reference_f1'])
            status['timing']=timing
            status['status']='evaluated'
            status['model_revision']=engine.revision
    # Il feedback e separato dai dati usati per monitorare.
    status['can_retrain']=bool(status['needs_retraining'] and Path('data/feedback.jsonl').exists())
    save_json('artifacts/monitor_status.json',status)
    output=os.getenv('GITHUB_OUTPUT')
    if output:
        with open(output,'a') as f: f.write('retrain='+str(status['can_retrain']).lower()+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__': main()
