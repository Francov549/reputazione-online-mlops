"""Pubblico solo gli artefatti esplicitamente selezionati, senza credenziali nei file."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

if __name__=='__main__':
    token=os.environ['HF_TOKEN']
    model_id=os.environ['HF_MODEL_ID']
    space_id=os.environ['HF_SPACE_ID']
    api=HfApi(token=token)
    report=json.loads(Path('artifacts/report.json').read_text())
    if report['test']['macro_f1']<.60:
        raise RuntimeError('Deploy fermato: metrica sotto la soglia minima')
    api.create_repo(model_id,repo_type='model',exist_ok=True)
    api.upload_folder(repo_id=model_id,folder_path='artifacts/champion',commit_message='Modello selezionato dalla pipeline')
    model_sha=api.model_info(model_id).sha
    api.create_repo(space_id,repo_type='space',space_sdk='gradio',exist_ok=True)
    for name in ['app.py','reputation.py','requirements.txt']:
        api.upload_file(path_or_fileobj=name,path_in_repo=name,repo_id=space_id,repo_type='space')
    readme='---\ntitle: Reputazione online\nemoji: 📊\ncolorFrom: blue\ncolorTo: green\nsdk: gradio\nsdk_version: 5.49.1\napp_file: app.py\npinned: false\n---\n\nProgetto di master. Testi in inglese, modello CardiffNLP.\n'
    api.upload_file(path_or_fileobj=readme.encode(),path_in_repo='README.md',repo_id=space_id,repo_type='space')
    api.add_space_variable(space_id,'MODEL_ID',model_id)
    api.add_space_variable(space_id,'MODEL_REVISION',model_sha)
    api.upload_file(path_or_fileobj='artifacts/report.json',path_in_repo='report.json',repo_id=model_id)
    print('Space:', 'https://huggingface.co/spaces/'+space_id)
