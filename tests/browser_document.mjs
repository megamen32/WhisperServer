// Run a real Chrome document until its browser scenario publishes a result.
import {spawn} from 'node:child_process';
import {existsSync,readFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const [chromePath,pagePath,profile]=process.argv.slice(2);
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket;let serial=0;const waiting=new Map();
const chrome=spawn(chromePath,['--headless=new','--no-sandbox','--disable-gpu',
 '--disable-dev-shm-usage','--disable-extensions','--disable-background-networking',
 '--disable-component-update','--no-proxy-server','--disable-quic','--disable-default-apps',
 '--disable-sync','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1',
 '--remote-debugging-port=0','--window-size=1440,1000','--user-data-dir='+profile,'about:blank'],
 {detached:true,stdio:'ignore'});
try {
 const deadline=Date.now()+25000;const portFile=join(profile,'DevToolsActivePort');
 while(!existsSync(portFile)){if(Date.now()>deadline)throw Error('Chrome debugging port did not open');await delay(50);}
 const port=readFileSync(portFile,'utf8').split('\n')[0];
 const pages=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json();
 socket=new WebSocket(pages.find(page=>page.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 socket.addEventListener('message',event=>{const m=JSON.parse(String(event.data));const p=waiting.get(m.id);if(!p)return;waiting.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(m.error.message)):p.resolve(m.result);});
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{waiting.delete(id);reject(Error('CDP timeout '+method));},10000);waiting.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});
 await call('Page.enable');await call('Runtime.enable');
 await call('Page.navigate',{url:pathToFileURL(pagePath).href});
 while(true){
  const r=await call('Runtime.evaluate',{expression:'JSON.stringify({result:document.body?.dataset.testResult,error:document.body?.dataset.scriptError})',returnByValue:true});
  const state=JSON.parse(r.result?.value||'{}');
  if(state.error)throw Error(state.error);
  if(state.result!==undefined){console.log(state.result);break;}
  if(Date.now()>deadline)throw Error('Browser scenario did not finish');
  await delay(50);
 }
} finally {
 socket?.close();for(const p of waiting.values()){clearTimeout(p.timer);}
 try{process.kill(-chrome.pid,'SIGKILL');}catch{}
}
