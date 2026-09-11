# Monitoraggio della reputazione online

Progetto di master di Simone Vitale.

**Consegna:** [notebook Google Colab](https://colab.research.google.com/drive/1EOhnBFk6H715dJWCQPu12EaM9TG2xlFc).

## Cosa ho realizzato

Ho usato **cardiffnlp/twitter-roberta-base-sentiment-latest**, richiesto dalla traccia, per classificare testi inglesi in negativo, neutro e positivo. Ho aggiunto valutazione, retraining della testa classificatrice, monitoraggio a finestre e pipeline GitHub Actions. Uso il nome MachineInnovators Inc., dato che nella traccia il nome non e uniforme.

Il progetto e una dimostrazione MLOps: i dati sono TweetEval, mentre i casi aziendali e le finestre temporali sono simulati. Non e un servizio gia collegato ai social dell'azienda.

## Risultati Colab

| Esperimento | N | Accuracy | Macro-F1 | Recall negativo |
|---|---:|---:|---:|---:|
| Baseline, validation | 600 | 0.7917 | 0.7856 | 0.8602 |
| Candidato, validation | 600 | 0.7817 | 0.7740 | 0.8172 |
| Modello selezionato, test | 1200 | 0.7375 | 0.7368 | 0.8196 |

Il candidato non e stato promosso: peggiora sulla validation. Il test finale riguarda il modello iniziale. I report riportano anche bootstrap, baseline maggioritaria, revisioni e ambiente effettivo. Cambiando hardware o dipendenze i risultati possono variare.

## Esecuzione

Nel notebook eseguo le celle in ordine. Fuori da Colab uso Python 3.12 in un ambiente virtuale pulito:

    pip install -r requirements.txt
    python -m pytest test_pipeline.py -q
    python reputation.py
    python app.py

Imposto RUN_MODEL_TESTS=1 per il test con il modello reale. Il codice usa GPU se disponibile, altrimenti CPU. Il modello richiede circa 500 MB. Congelo RoBERTa e aggiorno la testa per due epoche: e un compromesso didattico, non un fine-tuning completo.

## File

- reputation.py: dati, preprocessing, inferenza, metriche, retraining e selezione.
- test_pipeline.py: test di unita e integrazione con salvataggio e ricaricamento.
- monitor_batch.py: monitoraggio dei nuovi lotti.
- app.py: interfaccia Gradio; deploy.py: deploy facoltativo.
- .github/workflows/ci-cd.yml: test, monitoraggio e training condizionale.
- reports/: risultati reali dell'esecuzione Colab, senza ridistribuire il dataset.

## Metodo

Uso gli split ufficiali TweetEval sentiment. Tolgo vuoti e duplicati normalizzati, proteggendo prima test e validation, poi train. Il campionamento stratificato usa seed 42: 1200 train, 600 validation e 1200 test. Le revisioni usate sono in revisions.json. Mantengo negazioni, emoji e punteggiatura e normalizzo URL e menzioni.

Il modello e gia stato fine-tuned su TweetEval: questa e una verifica in-domain, non una prova su dati aziendali nuovi. Classi: 0=negativo, 1=neutro, 2=positivo. Uso macro-F1 per dare peso a tutte le classi, insieme ad accuracy e recall negativo.

## Monitoraggio e retraining

Distinguo quota negativa (reputazione), Jensen-Shannon delle predizioni (drift) e macro-F1 con etichette umane (qualita). La confidence non e calibrata. Soglie didattiche: minimo 30 testi, quota negativa >=50%, JS >0.10 e calo macro-F1 >0.05. Le soglie vanno calibrate su uno storico reale. Il drift o l'aumento del sentiment negativo non provano da soli il degrado del modello.

Nuovi dati: data/incoming.jsonl con un oggetto per riga e campo text; label (0/1/2) e human_verified=true se disponibili. Il retraining richiede anche data/feedback.jsonl con almeno 30 esempi nuovi annotati da persone, unici e separati da validation, test e lotto di monitoraggio. Non uso pseudo-label come verita.

GitHub Actions esegue i test a ogni push e pull request. Ogni lunedi alle 06:17 UTC e sui push al ramo principale controlla i nuovi lotti. Senza dati registra no_new_data e non avvia training. Con degrado misurato e feedback disponibile avvia il retraining. Per ripetere l'esperimento didattico uso Actions > Run workflow e seleziono run_training. La pianificazione GitHub puo subire ritardi o venire disattivata su repository inattivi.

Promuovo solo con macro-F1 >=0.60, guadagno >=0.005 e perdita recall negativo <=0.02 sulla validation. Il test non decide la promozione. Questi criteri non dimostrano significativita statistica: in produzione aggiungerei validation temporale recente e canary deployment.

## Hugging Face facoltativo e limiti

Il deploy e disattivato per default. Per abilitarlo servono secret HF_TOKEN e variabili HF_MODEL_ID, HF_SPACE_ID, ENABLE_HF_DEPLOY=true. Non inserisco token nei file. Dopo il deploy aggiorno CHAMPION_MODEL_ID e reports/reference.json con il modello e il report approvati. Lo Space usa una revisione precisa; per rollback imposto MODEL_REVISION al commit precedente.

Colab e gli Space gratuiti possono sospendersi. Per un servizio reale servono collector autorizzato, storage persistente e allarmi: qui non invio notifiche a persone. Altri limiti sono sarcasmo, contesto mancante, lingue diverse dall'inglese e lessico nuovo. Prima dell'uso operativo raccoglierei un campione aziendale annotato.

## Fonti

- [Model card CardiffNLP](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), inglese, CC BY 4.0.
- [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) e [repository originale](https://github.com/cardiffnlp/tweeteval). Non ripubblico il corpus; si applicano le condizioni delle fonti.
- [TimeLMs, Loureiro et al. 2022](https://aclanthology.org/2022.acl-demo.25/).
- [Deploy da GitHub Actions](https://huggingface.co/docs/hub/spaces-github-actions).
