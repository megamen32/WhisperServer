"""The real UI must expose same-settings retry and resume the new attempt's stream."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_quarantined_job_has_explicit_retry_without_duplicate_submission(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    node = shutil.which("node")
    if not chrome or not node:
        pytest.skip("Chrome and Node.js are required")
    template = (Path(__file__).parents[1] / "templates/webui.html").read_text()
    template = (template.replace("__DEFAULT_MODEL__", '"small"')
                .replace("__SUMMARY_CACHE_VERSION__", '"retry-test"')
                .replace("__WEBUI_CSRF_TOKEN__", '"test-csrf"'))
    fixture = r"""
    window.addEventListener('error',event=>document.body.dataset.scriptError=event.message);
    Object.defineProperty(window,'indexedDB',{value:undefined});
    window.retryCalls=0;window.streams=[];
    window.retryJob={id:'failed-job',filename:'saved.wav',status:'error',attempt:3,
      error:'Автоматический повтор остановлен',source_available:true,diarize:false,
      recovery:{quarantined:true,failures:3},result:{segments:[],text:''}};
    window.EventSource=class {
      constructor(url){this.url=url;this.readyState=1;streams.push(this);}
      close(){this.readyState=2;}
    };
    window.fetch=async(input,options={})=>{
      const url=String(input);let value={};
      if(url==='/models')value=['small'];
      else if(url==='/web/byok/presets')value=[];
      else if(url==='/web/session')value={token:'test-csrf'};
      else if(url==='/web/identity')value={has_recovery_key:false,job_count:1};
      else if(url==='/web/jobs/failed-job/retry'){
        retryCalls++;
        await new Promise(resolve=>window.releaseRetry=resolve);
        retryJob={...retryJob,status:'queued',attempt:4,error:null,recovery:null};
        value=retryJob;
      }
      else if(url==='/web/jobs')value={jobs:[retryJob]};
      else if(url==='/web/jobs/failed-job')value=retryJob;
      return new Response(JSON.stringify(value),{status:200,headers:{'Content-Type':'application/json'}});
    };
    """
    scenario = r"""
    (async()=>{
      const failures=[];const check=(ok,text)=>{if(!ok)failures.push(text);};
      const pause=()=>new Promise(resolve=>setTimeout(resolve,70));
      try {
        await initializationPromise;await selectJob('failed-job');await pause();
        const retry=document.querySelector('#retry-job');
        const historyRetry=document.querySelector('.retry-action');
        check(retry && !retry.hidden && !retry.disabled,'failed job has no visible retry control');
        check(Boolean(historyRetry),'history has no retry action');
        if(retry && historyRetry){
          retry.click();historyRetry.click();await pause();
          check(retryCalls===1,'two retry controls submitted duplicate requests');
          check(retry.disabled,'retry control remains enabled while request is pending');
          window.releaseRetry();await pause();await pause();
          check(selectedJob?.attempt===4 && selectedJob?.status==='queued','retry did not select the new attempt');
          check(retry.hidden && !cancelButton.hidden,'new attempt did not switch to cancellation controls');
          check(streams.some(source=>source.url.includes('/failed-job/events') && source.readyState===1),
            'retry did not resume incremental results');
        }
        retryJob={...retryJob,status:'error',source_available:false,error:'Source removed'};
        await loadHistory();renderJob(retryJob);await pause();
        check(!document.querySelector('.retry-action'),'source-free history still offers retry');
        check(!retry || retry.hidden,'source-free selected job still offers retry');
        await selectJob('');
      }catch(error){failures.push(error.stack||error.message);}
      document.body.dataset.testResult=JSON.stringify(failures);
    })();
    """
    template = template.replace("<script>", "<script>" + fixture + "</script><script>", 1)
    template = template.replace("</body>", "<script>" + scenario + "</script></body>")
    page = tmp_path / "retry.html"
    page.write_text(template)
    result = subprocess.run([node, str(Path(__file__).with_name("browser_document.mjs")),
                             chrome, str(page), str(tmp_path / "chrome")],
                            capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stderr[-2000:]
    assert not json.loads(result.stdout.strip()), result.stdout
