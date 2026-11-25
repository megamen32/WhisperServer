"""Browser controls keep recovery secrets out of persistent storage and preserve history."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_history_recovery_and_source_controls(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("Chromium is required")
    template = (Path(__file__).parents[1] / "templates/webui.html").read_text()
    template = (template.replace("__DEFAULT_MODEL__", '"small"')
                .replace("__SUMMARY_CACHE_VERSION__", '"identity-test"')
                .replace("__WEBUI_CSRF_TOKEN__", '"csrf-test"'))
    fixture = r"""
    Object.defineProperty(window,'indexedDB',{value:undefined});
    window.calls=[];window.keyIssued=false;window.merged=false;window.sourceRemoved=false;window.sourcePending=false;window.sourceFailures=0;
    window.confirm=()=>true;
    const saved={id:'saved-job',filename:'saved.wav',status:'done',attempt:1,diarize:false,
      transcript_id:'saved-job',result:{text:'Saved text',segments:[],transcript_id:'saved-job'},
      source_available:true};
    window.fetch=async(input,options={})=>{
      const url=String(input);calls.push({url,method:options.method||'GET'});
      let value={},status=200;
      if(url==='/models') value=['small'];
      else if(url==='/web/byok/presets') value=[];
      else if(url==='/web/session') value={token:'csrf-refreshed'};
      else if(url==='/web/identity') value={has_recovery_key:keyIssued,job_count:1};
      else if(url==='/web/identity/recovery-key') {keyIssued=true;value={recovery_key:'fixture-not-a-real-recovery-secret'};}
      else if(url==='/web/identity/recover') {
        const body=JSON.parse(options.body||'{}');
        if(!body.merge_current) {status=409;value={detail:{code:'merge_required',count:1}};}
        else {merged=true;value={ok:true,merged_jobs:1,token:'new-csrf'};}
      }
      else if(url==='/web/jobs/saved-job/source') {
        if(!sourceFailures++) {sourcePending=true;status=503;value={detail:'Fixture unlink failure'};}
        else {sourceRemoved=true;sourcePending=false;value={...saved,source_available:false,source_removal_pending:false};}
      }
      else if(url==='/web/jobs') value={jobs:[{...saved,source_available:!sourceRemoved&&!sourcePending,source_removal_pending:sourcePending}]};
      else if(url==='/web/jobs/saved-job') value={...saved,source_available:!sourceRemoved&&!sourcePending,source_removal_pending:sourcePending};
      return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});
    };
    """
    scenario = r"""
    (async()=>{
      const failures=[];const check=(ok,message)=>{if(!ok)failures.push(message);};
      const pause=()=>new Promise(resolve=>setTimeout(resolve,50));
      try {
        await initializationPromise;await pause();
        const issue=document.querySelector('#create-recovery-key');
        check(Boolean(issue),'recovery key control is missing');
        if(issue) {
          issue.click();await pause();
          const output=document.querySelector('#issued-recovery-key');
          check(output?.value.length>0 && output.type==='password','one-time key is not masked');
          document.querySelector('#hide-recovery-key')?.click();
          check(output.value==='','acknowledged key remained in DOM');
          const field=document.querySelector('#recover-history-key');
          field.value='fixture-not-a-real-recovery-secret';
          document.querySelector('#recover-history-form').dispatchEvent(new Event('submit',{cancelable:true}));
          await pause();await pause();
          check(merged,'explicit history merge did not complete');
          check(field.value==='','recovery input retained secret after request');
          check(historySelect.options.length===2,'recovery discarded history');
        }
        await loadHistory();
        const remove=document.querySelector('.source-action');
        check(Boolean(remove),'source-only action is missing');
        if(remove) {
          remove.click();await pause();await pause();
          const retry=document.querySelector('.source-action');
          check(retry?.textContent.includes('Повторить'),'failed unlink has no retry action');
          if(retry) {retry.click();await pause();await pause();}
        }
        check(sourceRemoved,'source-only action did not call its API');
        check(historySelect.options.length===2,'source removal deleted history');
        check(!calls.some(call=>call.url==='/web/jobs/saved-job'&&call.method==='DELETE'),
          'source action used destructive history deletion');
      } catch(error) {failures.push(error.name+': '+error.message);}
      document.body.dataset.testResult=JSON.stringify(failures);
    })();
    """
    template = template.replace("<script>", "<script>" + fixture + "</script><script>", 1)
    template = template.replace("</body>", "<script>" + scenario + "</script></body>")
    page = tmp_path / "identity.html"
    page.write_text(template)
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for Chrome CDP")
    result = subprocess.run([node, str(Path(__file__).with_name("browser_document.mjs")),
        chrome, str(page), str(tmp_path / "profile")], capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stderr[-2000:]
    failures = json.loads(result.stdout.strip())
    assert not failures, failures
