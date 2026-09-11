"""Piccola applicazione per provare il modello, predisposta per Hugging Face Spaces."""
import os
from functools import lru_cache
import gradio as gr
from reputation import SentimentModel, MODEL_ID, LABELS

@lru_cache(maxsize=1)
def get_engine():
    return SentimentModel(os.getenv('MODEL_ID',MODEL_ID),os.getenv('MODEL_REVISION') or None)

def analyze(text):
    if not text or not text.strip():
        raise gr.Error('Inserisci un testo in inglese.')
    if len(text)>5000:
        raise gr.Error('Per la demo uso al massimo 5.000 caratteri.')
    p,timing=get_engine().predict([text])
    # La confidence non e una probabilita calibrata di avere ragione.
    return dict(zip(LABELS,map(float,p[0]))),f"Tempo: {timing['seconds']:.3f} s. Testo troncato a 128 token se necessario."

def build_app():
    return gr.Interface(fn=analyze,inputs=gr.Textbox(label='Messaggio in inglese',lines=4),
        outputs=[gr.Label(label='Sentiment'),gr.Textbox(label='Informazioni')],
        title='Monitoraggio della reputazione online',
        description='Progetto di master. Analizzo testi in inglese con Twitter-RoBERTa. Non salvo il testo inserito. La demo non raccoglie automaticamente messaggi social.',
        examples=['The support team was helpful and solved my issue.','The product update is scheduled for Monday.','The service keeps failing and I am disappointed.'])

if __name__=='__main__':
    build_app().launch(server_name='0.0.0.0',server_port=7860)
