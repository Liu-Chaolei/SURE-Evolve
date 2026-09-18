"""JSON-only snapshots of explicitly registered search state types."""
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import time


class SearchCheckpoint:
    def __init__(self, path: Path, identity: str, registry: dict):
        self.path, self.identity, self.registry = path, identity, registry

    def encode(self, value):
        if isinstance(value, Enum):
            return {'enum': type(value).__name__, 'value': value.value}
        if is_dataclass(value) and not isinstance(value, type):
            return {'type': type(value).__name__, 'fields': {f.name:self.encode(getattr(value,f.name)) for f in fields(value) if f.init}}
        if isinstance(value, dict):
            return {'dict': [[self.encode(k),self.encode(v)] for k,v in value.items()]}
        if isinstance(value, (tuple, set, list, frozenset)):
            return {type(value).__name__: [self.encode(v) for v in value]}
        if value is None or isinstance(value,(str,int,float,bool)):
            return value
        raise TypeError('Unsupported search checkpoint type: '+type(value).__name__)

    def decode(self, value):
        if not isinstance(value,dict): return value
        if 'enum' in value: return self.registry[value['enum']](value['value'])
        if 'type' in value:
            return self.registry[value['type']](**{k:self.decode(v) for k,v in value['fields'].items()})
        if 'dict' in value: return {self.decode(k):self.decode(v) for k,v in value['dict']}
        for name,kind in [('tuple',tuple),('set',set),('list',list),('frozenset',frozenset)]:
            if name in value:return kind(self.decode(v) for v in value[name])
        raise ValueError('Unknown search checkpoint representation')

    def load(self):
        if not self.path.exists():return None
        data=json.loads(self.path.read_text())
        if data['schema']!='xlab.search.iteration.v1' or data['identity']!=self.identity:
            raise ValueError('Search iteration checkpoint identity changed')
        raw=json.dumps(data['state'],sort_keys=True,allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest()!=data['state_digest']:
            raise ValueError('Search iteration checkpoint digest mismatch')
        return self.decode(data['state'])

    def save(self, state, *, status='running'):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        encoded=self.encode(state)
        raw=json.dumps(encoded,sort_keys=True,allow_nan=False).encode()
        doc={'schema':'xlab.search.iteration.v1','identity':self.identity,
             'state':encoded,'state_digest':hashlib.sha256(raw).hexdigest(),
             'status':status,'updated_at':time.time()}
        counters = state['counters']
        doc['progress'] = {'iterations':counters.iterations,'children':counters.children,
                           'generation_calls':counters.generation_calls,'evaluator_calls':counters.evaluator_calls}
        pending=self.path.with_suffix('.pending')
        pending.write_text(json.dumps(doc,sort_keys=True,allow_nan=False)+'\n')
        pending.replace(self.path)
