"""HTTP API, сторінка для демонстрації і моніторинг.

    uvicorn finmon.api:app --reload

/            — демо: обрати спрацювання і отримати висновок агентів
/monitor     — моніторинг: помилки, час, вартість, рішення, переключення провайдерів
/metrics     — метрики у форматі Prometheus
/stats       — те саме, що /monitor, у JSON
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import make_asgi_app

from . import obs
from .agents.graph import Copilot
from .config import AGENT_CONFIGS, DEFAULT_CONFIG, MODELS
from .llm import ModelRefused, ProviderUnavailable

app = FastAPI(title="FinMon Copilot", version="0.2.0",
              description="Мультиагентний помічник аналітика фінмоніторингу: LangGraph + Claude і Gemini + RAG по Закону 361-IX і Положенню НБУ №65.")
app.mount("/metrics", make_asgi_app())
copilot = Copilot("eval")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "alerts": len(copilot.bank.alerts), "default_config": DEFAULT_CONFIG}


@app.get("/alerts")
def alerts() -> list[dict]:
    out = []
    for a in copilot.bank.alerts.values():
        c = copilot.bank.clients[a["client_id"]]
        out.append({"alert_id": a["alert_id"], "rule_id": a["rule_id"], "rule_text": a["rule_text"], "client": c["name"], "client_type": c["type"]})
    return out


@app.get("/configs")
def configs() -> dict:
    return {name: {role: MODELS[m] for role, m in roles.items()} for name, roles in AGENT_CONFIGS.items()}


@app.post("/alerts/{alert_id}/investigate")
def investigate(alert_id: str, config: str = DEFAULT_CONFIG) -> dict:
    if alert_id not in copilot.bank.alerts:
        raise HTTPException(404, "Спрацювання не знайдено")
    if config not in AGENT_CONFIGS:
        raise HTTPException(400, f"config має бути одним з: {', '.join(AGENT_CONFIGS)}")
    try:
        res = copilot.run(alert_id, config)
    except ProviderUnavailable as e:  # і основний, і резервний провайдер недоступні
        raise HTTPException(503, f"Моделі тимчасово недоступні: {e}") from e
    except ModelRefused as e:
        raise HTTPException(422, f"Модель відмовилась розбирати спрацювання: {e}") from e
    return {**res["final"], "trace": {"llm_calls": res["usage"], "tool_calls": res["tool_log"]}}


@app.get("/stats")
def stats() -> dict:
    return obs.RECENT.summary()


PAGE = """<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinMon Copilot</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f5f4;color:#161616}
main{max-width:1000px;margin:0 auto;padding:24px 16px;display:grid;grid-template-columns:320px 1fr;gap:16px}
h1{font-size:20px;margin:0 0 4px} .muted{color:#666;font-size:13px} a{color:#2450c9}
ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px;max-height:80vh;overflow:auto}
li button{width:100%;text-align:left;border:1px solid #ddd;background:#fff;border-radius:8px;padding:8px 10px;cursor:pointer;font:inherit}
li button:hover{border-color:#111}
#out{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;min-height:200px;white-space:pre-wrap;font-size:14px}
.d{font-weight:700;font-size:18px}.ESCALATE{color:#b3261e}.CLOSE{color:#1d7a4a}.REQUEST_INFO{color:#9a6200}
select{font:inherit;padding:4px}
@media(max-width:760px){main{grid-template-columns:1fr}}
</style>
<main><div><h1>FinMon Copilot</h1><p class="muted">Синтетичні дані. Оберіть спрацювання. <a href="/monitor">Моніторинг</a></p>
<p>Модель: <select id="cfg"><option>routed</option><option>haiku</option><option>sonnet</option><option>opus</option><option>gemini</option><option>gemini_lite</option></select></p><ul id="list"></ul></div>
<div id="out">Тут з'явиться висновок агентів.</div></main>
<script>
const out=document.getElementById('out');
fetch('/alerts').then(r=>r.json()).then(a=>{const ul=document.getElementById('list');a.forEach(x=>{const li=document.createElement('li');const b=document.createElement('button');
b.textContent=x.alert_id+' · '+x.rule_id+' · '+x.client;b.title=x.rule_text;b.onclick=()=>run(x.alert_id);li.appendChild(b);ul.appendChild(li);});});
async function run(id){out.textContent='Агенти працюють… (30–90 с)';const cfg=document.getElementById('cfg').value;
const r=await fetch('/alerts/'+id+'/investigate?config='+cfg,{method:'POST'});const j=await r.json();
if(!r.ok){out.textContent='Помилка '+r.status+': '+(j.detail||'');return;}
out.innerHTML='';const d=document.createElement('div');d.className='d '+j.decision;d.textContent=j.decision+' · впевненість '+j.confidence;out.appendChild(d);
const t=document.createElement('div');t.textContent='\\n'+j.summary_ua+'\\n\\nПравова основа:\\n'+j.legal_basis.map(b=>'• '+b.source+' — '+b.why).join('\\n')+
'\\n\\nНаступні кроки:\\n'+j.next_steps.map(s=>'• '+s).join('\\n')+'\\n\\n$'+j.cost_usd+' · '+j.latency_s+' с · інструментів: '+j.tool_calls+' · моделі: '+j.models_used.join(', ')+
(j.provider_fallbacks?' · переключень на резерв: '+j.provider_fallbacks:'');out.appendChild(t);}
</script></html>"""

MONITOR = """<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinMon · моніторинг</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f5f4;color:#161616}
main{max-width:1000px;margin:0 auto;padding:24px 16px;display:flex;flex-direction:column;gap:16px}
h1{font-size:20px;margin:0} .muted{color:#666;font-size:13px} a{color:#2450c9}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.tile span{font-size:13px;color:#666}
.bad b{color:#b3261e}.ok b{color:#1d7a4a}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;border-radius:10px;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums}th{font-size:12px;color:#666}
.wrap{overflow-x:auto}
</style>
<main><div><h1>Моніторинг FinMon Copilot</h1><p class="muted">Останні розбори в цьому процесі. Оновлюється кожні 5 с. Метрики для Prometheus: <a href="/metrics">/metrics</a> · <a href="/">демо</a></p></div>
<div class="tiles" id="tiles"></div>
<h2 style="font-size:16px;margin:0">Останні розбори</h2><div class="wrap"><table><thead><tr><th>Час</th><th>Спрацювання</th><th>Конфігурація</th><th>Статус</th><th>Рішення</th><th>Час, с</th><th>$</th><th>Резерв</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2 style="font-size:16px;margin:0">Переключення на резервного провайдера</h2><div class="wrap"><table><thead><tr><th>Час</th><th>Спрацювання</th><th>Вузол</th><th>З</th><th>На</th><th>Причина</th></tr></thead><tbody id="fb"></tbody></table></div>
</main>
<script>
function td(v){const e=document.createElement('td');e.textContent=v==null?'—':v;return e}
async function load(){const s=await (await fetch('/stats')).json();
const tiles=[['Розборів',s.investigations,''],['Помилки',(s.error_rate*100).toFixed(1)+'%',s.error_rate>0.05?'bad':'ok'],['Час p50, с',s.latency_p50_s,''],['Час p95, с',s.latency_p95_s,s.latency_p95_s>120?'bad':''],
['$ за розбір',s.cost_avg_usd,s.cost_avg_usd>0.1?'bad':''],['Порушення посилань',s.grounding_violations,s.grounding_violations?'bad':'ok'],['Переключень на резерв',s.fallbacks,s.fallbacks?'bad':'ok'],
['Рішення',Object.entries(s.decisions).map(([k,v])=>k+' '+v).join(' · ')||'—','']];
const T=document.getElementById('tiles');T.innerHTML='';tiles.forEach(([n,v,c])=>{const d=document.createElement('div');d.className='tile '+c;const b=document.createElement('b');b.textContent=v==null?'—':v;const sp=document.createElement('span');sp.textContent=n;d.append(b,sp);T.append(d)});
const R=document.getElementById('rows');R.innerHTML='';s.recent.forEach(i=>{const tr=document.createElement('tr');[i.ts,i.alert_id,i.config,i.status==='ok'?'ok':i.error,i.decision,i.latency_s,i.cost_usd,i.fallbacks].forEach(v=>tr.append(td(v)));R.append(tr)});
const F=document.getElementById('fb');F.innerHTML='';s.last_fallbacks.slice().reverse().forEach(i=>{const tr=document.createElement('tr');[i.ts,i.alert_id,i.node,i.from,i.to,i.reason].forEach(v=>tr.append(td(v)));F.append(tr)});}
load();setInterval(load,5000);
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/monitor", response_class=HTMLResponse)
def monitor() -> str:
    return MONITOR
