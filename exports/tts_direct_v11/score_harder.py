import concurrent.futures,json,subprocess,sys
from pathlib import Path
root=Path('/shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11')
source='/shared/chaolei.liu/SURE-Evolve/runs/f5tts_premium_evolution_20260914_v11_direct/source/a366f1f0bfc270bb97535b8b205ff9ec13f97377040a63a207f19c7ed559c395'
def one(w):
 req=w/'score_request.json'; out=w/'score_result.json'; log=(w/'scoring.log').open('a')
 try:
  subprocess.run(['python',source+'/playground/sure_master/tools/score_task.py',str(req),str(out)],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
  return True
 except Exception as e:
  (w/'score_error.txt').write_text(str(e));return False
for kind in ('hardcase_20260918_r2','nonpara_20260918_r2'):
 ws=[p.parent for p in (root/kind).glob('*/score_request.json')]
 print(kind,len(ws),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  print(list(ex.map(one,ws)),flush=True)
