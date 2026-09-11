"""Controllo i casi limite e il comportamento della pipeline."""
import os
import numpy as np
import pytest
from reputation import normalize, metrics, promotion, monitor, js_divergence, digest

@pytest.mark.parametrize('value',[None,'', '   ',42])
def test_invalid_text(value):
    with pytest.raises(ValueError): normalize(value)

def test_normalization_preserves_sentiment():
    assert normalize('@alice I am NOT happy 😡 https://example.org')=='@user I am NOT happy 😡 http'
    assert digest('@alice hello')==digest('@bob hello')

def test_metrics_missing_class_is_penalized():
    result=metrics([0,1,2],[1,1,1])
    assert result['accuracy']==pytest.approx(1/3)
    assert result['negative_recall']==0
    assert result['macro_f1']<result['accuracy']

def test_promotion_needs_improvement_and_negative_recall():
    base={'macro_f1':.75,'negative_recall':.80}
    assert promotion(base,{'macro_f1':.76,'negative_recall':.80})
    assert not promotion(base,{'macro_f1':.74,'negative_recall':.85})
    assert not promotion(base,{'macro_f1':.80,'negative_recall':.70})
    assert not promotion(base,{'macro_f1':.75,'negative_recall':.80})

def test_drift_symmetric_and_bounded():
    assert js_divergence([.3,.4,.3],[.3,.4,.3])==pytest.approx(0)
    x=js_divergence([1,0,0],[0,0,1])
    assert .99<x<=1
    assert x==pytest.approx(js_divergence([0,0,1],[1,0,0]))

def test_negative_sentiment_does_not_trigger_retraining_without_labels():
    result=monitor([.2,.5,.3],np.tile([.9,.05,.05],(50,1)))
    assert result['reputation_alert']
    assert result['prediction_drift_alert']
    assert not result['needs_retraining']

def test_degradation_requires_minimum_window():
    bad=np.tile([.9,.05,.05],(30,1))
    assert monitor([.3,.4,.3],bad,[1]*30,.75)['needs_retraining']
    assert not monitor([.3,.4,.3],bad[:5],[1]*5,.75)['needs_retraining']

@pytest.mark.parametrize('probs',[[],[[1,2]],[[.2,.2,.2]],[[float('nan'),0,1]]])
def test_invalid_probabilities(probs):
    with pytest.raises(ValueError): monitor([.3,.4,.3],probs)

@pytest.mark.skipif(os.getenv('RUN_MODEL_TESTS')!='1',reason='Test integrazione con download del modello reale')
def test_real_model_and_reload(tmp_path):
    from reputation import SentimentModel
    engine=SentimentModel()
    texts=['I love this excellent service!','The service is awful and useless.','The meeting starts at three.']
    p,_=engine.predict(texts)
    assert p.shape==(3,3)
    assert np.allclose(p.sum(1),1,atol=1e-5)
    assert p[0].argmax()==2 and p[1].argmax()==0
    engine.save(tmp_path/'model')
    reloaded=SentimentModel(str(tmp_path/'model'))
    q,_=reloaded.predict(texts)
    assert np.allclose(p,q,atol=1e-5)
