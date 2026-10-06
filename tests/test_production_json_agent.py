"""Native executable JSON contracts; no mock provider or fabricated serving result."""
import copy
import importlib.util
from pathlib import Path

import pytest

from graph_core.schema import Graph
from production import json_agent_adapter as adapter
from production import agent_adapter, conversation_adapter
from production.pipeline import ProductionError
from agent_helpers import Lab

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('json_serving_fixture',ROOT/'examples/make_json_serving_fixture.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_native_json_contract_preserves_graph_schema_and_legacy_implementation_identity():
    graph=fixture.graph();before=graph.to_json()
    text_pins=agent_adapter.implementation();conversation_pins=conversation_adapter.implementation()
    state,cfg=adapter.contract(graph,{'question':'SYNTHETIC teaching input'})
    assert cfg.output_field=='result' and state.field('result').type=='object'
    assert cfg.on_failure=='fail' and cfg.retry.maxRetries==0
    assert graph.to_json()==before
    pins=adapter.implementation()
    assert all(pins[p]==sha for p,sha in text_pins.items())
    assert agent_adapter.implementation()==text_pins and conversation_adapter.implementation()==conversation_pins
    assert 'production/json_agent_adapter.py' in pins and 'production/json_agent_adapter.py' not in text_pins


@pytest.mark.parametrize('change,code',[
    ({'on_failure':'route'},'E_AGENT_JSON_FAILURE'),
    ({'retry':{'maxRetries':1}},'E_AGENT_JSON_FAILURE'),
    ({'model':{'provider':'fixture','fixture':{'default':'{"colour":"red","count":3}'}}},'E_AGENT_PROVIDER'),
    ({'model':{'provider':'ollama','model':'qwen3.5:2b','think':True,'max_tokens':96}},'E_AGENT_PROVIDER'),
])
def test_native_json_contract_refuses_unbounded_failure_and_non_native_serving_provider(change,code):
    data=copy.deepcopy(fixture.graph().to_json());data['nodes'][1]['config'].update(change)
    with pytest.raises(ProductionError) as error:adapter.contract(Graph.model_validate(data),{'question':'SYNTHETIC teaching input'})
    assert error.value.code==code


@pytest.mark.parametrize('value',[
    {}, {'colour':'red','count':True}, {'colour':'green','count':3}, {'colour':'blue','count':float('nan')},
    {'colour':'red','count':3,'unexpected':1}, {'colour':'red','count':3,'extra':'x'*9000}, ['not','an','object'],
])
def test_output_uses_native_schema_refusals_and_finite_bounded_json(value):
    _,cfg=adapter.contract(fixture.graph(),{'question':'SYNTHETIC teaching input'})
    with pytest.raises(ProductionError) as error:adapter.output_value(value,cfg)
    assert error.value.code=='E_AGENT_JSON_OUTPUT'


def test_native_declared_output_validation_is_not_a_model_prediction():
    _,cfg=adapter.contract(fixture.graph(),{'question':'SYNTHETIC teaching input'})
    declared={'colour':'blue','count':7}
    assert adapter.output_value(declared,cfg)==declared


def test_declared_default_object_without_native_model_call_cannot_be_registered(tmp_path):
    data=copy.deepcopy(fixture.graph().to_json())
    data['agent']['state'].append({'name':'call_model','type':'boolean','default':False,'reducer':{'kind':'replace'},'scope':'turn'})
    next(f for f in data['agent']['state'] if f['name']=='result')['default']={'colour':'red','count':3}
    data['edges']=[e for e in data['edges'] if e['from']['node']!='prompt']
    data['agent']['routes']=[{'id':'conditional_model','from':'prompt','cases':[{'id':'call','when':{'field':'call_model','op':'is_true'},'to':'extract'}],'default':'END'}]
    graph=Graph.model_validate(data)
    lab=Lab(tmp_path);status,run=lab.run(graph,inp={'question':'SYNTHETIC declared default; model is never invoked'})
    assert status=='completed' and lab.final(run)['result']=={'colour':'red','count':3}
    assert lab.contexts(run)==[]
    assert adapter.is_candidate(lab.store,lab.store.get_run(run)) is None
    with pytest.raises(ProductionError) as error:adapter.build_manifest(lab.store,run,adapter.NODE)
    assert error.value.code=='E_AGENT_SOURCE'
