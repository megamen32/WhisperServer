"""Bounded browser rendering must retain full-document search and copying."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_long_transcript_keeps_bounded_dom_and_full_search_copy(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("Chromium is required for the long transcript regression")
    template = (Path(__file__).resolve().parents[1] / "templates/webui.html").read_text()
    template = (template.replace("__DEFAULT_MODEL__", '"gigaam-v3"')
                .replace("__SUMMARY_CACHE_VERSION__", '"long-test"')
                .replace("__WEBUI_CSRF_TOKEN__", '"test-csrf"'))
    fixtures = r"""
    window.addEventListener('error', event => document.body.dataset.scriptError = event.message);
    Object.defineProperty(window, 'indexedDB', {value:undefined});
    Object.defineProperty(navigator, 'clipboard', {value:{writeText:async value => window.copiedText=value}});
    window.longSegments=Array.from({length:4000},(_,i)=>({start:i*20,end:(i+1)*20,
      text:(i===3999?'ПОСЛЕДНИЙ уникальный текст':i===0?'НАЧАЛЬНЫЙ текст':'Фрагмент '+i)+
        (i%2?' короткий.':' Очень длинное предложение.'.repeat(18))+
        ([3,1800,3999].includes(i)?' точныйпоиск':''),
      speaker:i%2?'SPEAKER_01':'SPEAKER_00'}));
    window.longJob={id:'long-test',filename:'long.wav',status:'done',attempt:1,
      result:{segments:longSegments,text:longSegments.map(s=>s.text).join(' '),transcript_id:'long-transcript'},
      stream_cursor:'1:0:4000',progress:{phase:'done',percent:100}};
    window.fetch=async input=>{
      const url=String(input);let value={};
      if(url==='/models') value={models:['gigaam-v3']};
      else if(url==='/web/byok/presets') value=[];
      else if(url==='/web/session') value={token:'test-csrf'};
      else if(url==='/web/jobs') value={jobs:[]};
      else if(url.startsWith('/web/jobs/')) value=longJob;
      return new Response(JSON.stringify(value),{status:200,headers:{'Content-Type':'application/json'}});
    };
    """
    scenario = r"""
    (async()=>{
      const failures=[];
      const check=(ok,label)=>{if(!ok)failures.push(label)};
      try {
        await initializationPromise;
        await selectJob('long-test');
        const count=()=>transcriptContainer.querySelectorAll('.segment').length;
        check(count()<=160,'long transcript mounts '+count()+' segments instead of a bounded window');
        check(segmentsProcessed===4000,'virtual window changed total segment count');
        check(transcriptContainer.textContent.includes('НАЧАЛЬНЫЙ'),'initial window lost beginning');
        const scroller=transcriptContainer.closest('.document');
        scroller.style.scrollBehavior='auto';
        scroller.scrollTop=scroller.scrollHeight;
        scroller.dispatchEvent(new Event('scroll'));
        await new Promise(resolve=>setTimeout(resolve,100));
        check(transcriptContainer.textContent.includes('ПОСЛЕДНИЙ'),'scrolling to bottom did not render last segment');
        check(count()<=160,'scrolling expanded DOM without bound');
        transcriptSearch.value='точныйпоиск';refreshSearch();
        check(matchingSegments.length===3,'search did not include offscreen matches');
        moveToMatch(1);moveToMatch(1);
        check(transcriptContainer.querySelector('.current-match')?.textContent.includes('ПОСЛЕДНИЙ'),
          'match navigation did not reach offscreen last match');
        check(transcriptContainer.querySelector('mark')?.textContent==='точныйпоиск','search highlight is missing');
        copyResultButton.click();await new Promise(resolve=>setTimeout(resolve,30));
        check(window.copiedText?.includes('НАЧАЛЬНЫЙ') && window.copiedText?.includes('ПОСЛЕДНИЙ')
          && window.copiedText?.includes('Фрагмент 1900'),'copy omitted offscreen or filtered text');
        check(window.copiedText?.includes('Спикер 2:'),'copy lost speaker labels');
        transcriptSearch.value='';refreshSearch();
        check(count()<=160,'clearing search expanded DOM without bound');
        const scrollBefore=scroller.scrollTop;
        longJob={...longJob,status:'processing',progress:{phase:'transcribing',percent:99}};
        selectedJob.status='processing';
        renderJob(longJob);
        check(Math.abs(scroller.scrollTop-scrollBefore)<5,'duplicate snapshot moved the scroll anchor');
        await selectJob('');
        check(segmentsProcessed===0 && count()===0,'changing job retained old rendered window');
      } catch(error) { failures.push(error.stack || error.message); }
      document.body.dataset.testResult=JSON.stringify(failures);
    })();
    """
    template = template.replace("<script>", "<script>"+fixtures+"</script><script>",1)
    template = template.replace("</body>", "<script>"+scenario+"</script></body>")
    page=tmp_path/"long.html";page.write_text(template)
    node=shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for Chrome CDP")
    result=subprocess.run([node,str(Path(__file__).with_name("browser_document.mjs")),chrome,str(page),str(tmp_path/"chrome")],
        capture_output=True,text=True,timeout=35)
    assert result.returncode==0,result.stderr[-2000:]
    failures=json.loads(result.stdout.strip())
    assert not failures,failures
