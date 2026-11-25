"""Browser regression for partial job text, SSE recovery and upload byte progress."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_webui_streams_partial_jobs_and_upload_bytes(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("Chromium is required for the Web UI streaming regression")
    template = (Path(__file__).resolve().parents[1] / "templates/webui.html").read_text()
    template = (template.replace("__DEFAULT_MODEL__", '"gigaam-v3"')
                .replace("__SUMMARY_CACHE_VERSION__", '"stream-test"')
                .replace("__WEBUI_CSRF_TOKEN__", '"test-csrf"'))
    fixtures = r"""
    window.addEventListener('error', event => document.body.dataset.scriptError = event.message);
    Object.defineProperty(window, 'indexedDB', {value:undefined});
    window.testEvents = [];
    window.testJob = {id:'stream-test', filename:'test.wav', status:'processing', attempt:1,
        diarize:true, result:{segments:[],text:''},
        progress:{phase:'transcribing',percent:0,processed_seconds:0,total_seconds:60,segments:0}};
    window.fetch = async (input) => {
        const url = String(input);
        let value = {};
        if (url === '/models') value = {models:['gigaam-v3']};
        else if (url === '/web/byok/presets') value = [];
        else if (url === '/web/jobs') value = {jobs:[]};
        else if (url.startsWith('/web/jobs/')) value = testJob;
        return new Response(JSON.stringify(value), {status:200,headers:{'Content-Type':'application/json'}});
    };
    window.EventSource = class {
        constructor(url) { this.url=url; this.readyState=1; this.closed=false; testEvents.push(this); }
        close() { this.closed=true; this.readyState=2; }
        emit(value) { if (!this.closed) this.onmessage?.({data:JSON.stringify(value)}); }
    };
    window.testUploadSamples = [];
    window.XMLHttpRequest = class {
        constructor() { this.upload={}; this.status=200; this.responseText='{"offset":200}'; }
        open(method,url) { this.url=url; }
        setRequestHeader() {}
        getResponseHeader(name) { return name === 'X-Whisper-Next-CSRF' ? 'rotated-test-csrf' : null; }
        send() { this.upload.onprogress?.({loaded:100,total:200,lengthComputable:true});
                 this.upload.onprogress?.({loaded:200,total:200,lengthComputable:true});
                 this.onload?.(); }
    };
    """
    scenario = r"""
    (async () => {
      const failures=[];
      const check=(ok,label)=>{if(!ok) failures.push(label);};
      try {
        await initializationPromise;
        await selectJob('stream-test');
        const source=testEvents.at(-1);
        check(Boolean(source), 'SSE subscription was not opened');
        const first={start:0,end:20,text:'Первый фрагмент',speaker:'SPEAKER_00'};
        testJob={...testJob,result:{segments:[first],text:first.text},
          progress:{phase:'transcribing',percent:33,processed_seconds:20,total_seconds:60,segments:1}};
        if(source) source.emit(testJob); else renderJob(testJob);
        check(document.querySelectorAll('#transcript .segment').length === 1,
          'partial transcription is not shown while processing');
        check(progressBar.style.width === '33%', 'actual progress was not rendered');
        check(summarizeButton.disabled, 'partial transcription must not be summarized');
        const firstNode=transcriptContainer.firstElementChild;
        transcriptSearch.value='Первый';refreshSearch();
        const previousMatch=matchingSegments[0];
        if(source) source.emit(testJob); else renderJob(testJob);
        check(transcriptContainer.firstElementChild === firstNode, 'duplicate snapshot replaced existing text');
        const second={start:20,end:40,text:'Второй фрагмент',speaker:'SPEAKER_00'};
        testJob={...testJob,result:{segments:[first,second],text:'Первый фрагмент Второй фрагмент'},
          progress:{phase:'transcribing',percent:66,processed_seconds:40,total_seconds:60,segments:2}};
        if(source) source.emit(testJob); else renderJob(testJob);
        check(document.querySelectorAll('#transcript .segment').length === 2, 'new segment was not appended exactly once');
        check(matchingSegments[0] === previousMatch, 'stream update lost the current search result');
        testJob={...testJob,status:'done',result:{...testJob.result,transcript_id:'test-transcript'},
          progress:{phase:'done',percent:100,processed_seconds:60,total_seconds:60,segments:2}};
        if(source) source.emit(testJob); else renderJob(testJob);
        check(document.querySelectorAll('#transcript .segment').length === 2, 'final snapshot duplicated text');
        check(progressBar.style.width === '100%', 'completion did not reach 100 percent');
        check(!summarizeButton.disabled, 'completed transcription cannot be summarized');
        check(!source || source.closed, 'terminal job left SSE open');
        await api('/web/jobs/stream-test/upload?offset=0', {method:'PUT',
          headers:{'Content-Type':'application/octet-stream'}, body:new Blob(['test']),
          onUploadProgress:(event)=>testUploadSamples.push(event.loaded)});
        check(testUploadSamples.join(',') === '100,200', 'upload bytes are only shown after each chunk');
        check(webuiCsrfToken === 'rotated-test-csrf', 'upload lost the rotated CSRF token');
        testJob={...testJob,id:'reconnect-test',status:'processing',attempt:1,
          result:{segments:[first],text:first.text},
          progress:{phase:'transcribing',percent:33,processed_seconds:20,total_seconds:60,segments:1}};
        await selectJob('reconnect-test');
        const reconnectSource=testEvents.at(-1);
        testJob={...testJob,result:{segments:[first,second],text:'Первый фрагмент Второй фрагмент'},
          progress:{phase:'transcribing',percent:66,processed_seconds:40,total_seconds:60,segments:2}};
        reconnectSource?.onerror?.();
        await new Promise(resolve=>setTimeout(resolve,1100));
        check(document.querySelectorAll('#transcript .segment').length === 2,
          'SSE failure did not recover partial text through GET');
        const previousJob=testJob;
        testJob={...testJob,id:'other-test',result:{segments:[second],text:second.text}};
        await selectJob('other-test');
        check(!reconnectSource || reconnectSource.closed, 'switching jobs left the previous stream open');
        reconnectSource?.onmessage?.({data:JSON.stringify(previousJob)});
        check(selectedJobId === 'other-test' && transcriptContainer.textContent.includes('Второй')
          && !transcriptContainer.textContent.includes('Первый'), 'late event replaced the newly selected job');
        const retrySource=testEvents.at(-1);
        testJob={...testJob,attempt:2,result:{segments:[],text:''},
          progress:{phase:'loading',percent:null,processed_seconds:0,total_seconds:null,segments:0}};
        retrySource?.emit(testJob);
        check(document.querySelectorAll('#transcript .segment').length === 0, 'new attempt retained old partial text');
        check(progressBar.classList.contains('indeterminate'), 'preparation stage shows fake numeric progress');
        check(!retrySource || retrySource.closed, 'new attempt left the old stream open');
        retrySource?.onmessage?.({data:JSON.stringify({...previousJob,id:'other-test'})});
        check(selectedJob.attempt === 2 && document.querySelectorAll('#transcript .segment').length === 0,
          'late attempt restored stale text');
        await selectJob('');
      } catch(error) { failures.push(error.stack || error.message); }
      document.body.dataset.testResult=JSON.stringify(failures);
    })();
    """
    template = template.replace("<script>", "<script>" + fixtures + "</script><script>", 1)
    template = template.replace("</body>", "<script>" + scenario + "</script></body>")
    page = tmp_path / "streaming.html"
    page.write_text(template)
    result = subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check",
        f"--user-data-dir={tmp_path / 'chrome'}", "--virtual-time-budget=3000", "--dump-dom", page.as_uri()],
        capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr[-2000:]
    match = re.search(r'data-test-result="([^"]*)"', result.stdout)
    assert match, "browser scenario did not finish: " + result.stdout[result.stdout.find("<body"):result.stdout.find("<body")+500]
    import html
    failures = json.loads(html.unescape(match.group(1)))
    assert not failures, failures
