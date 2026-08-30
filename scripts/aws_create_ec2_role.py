#!/usr/bin/env python3
"""Create the EC2 instance role that lets the host pull the image from ECR."""
import sys
import time

sys.path.insert(0, "scripts")
from ffsession import Session

ROLE = "ai-defense-lab-ec2"
POLICY = "AmazonEC2ContainerRegistryReadOnly"
s = Session()

s.goto("https://us-east-1.console.aws.amazon.com/iam/home#/roles/create?step=selectEntities", settle=20)
print("URL:", s.url[:120])

# Step 1: trusted entity = AWS service, use case = EC2
print("service radio:", (s.click_radio("AWS service") or "")[:40])
time.sleep(3)
sel = s.script("""
// pick EC2 in the use-case list (radio or select)
for (const inp of document.querySelectorAll('input[type=radio]')) {
  const w = inp.closest('label,div,fieldset');
  const t = w ? (w.textContent || '') : '';
  if (/\\bEC2\\b/.test(t) && !/Image Builder|Instance Connect|Fleet|Spot|Auto Scaling/i.test(t)) {
    inp.scrollIntoView({block:'center'}); inp.click(); return t.trim().slice(0,70);
  }
}
// dropdown fallback
const box=[...document.querySelectorAll('button,input')].find(e=>/use case|choose a service/i.test((e.textContent||'')+(e.placeholder||'')));
return box ? 'dropdown:' + (box.textContent||box.placeholder) : 'ec2 use case not found';
""")
print("use case:", sel)
time.sleep(2)
s.script("""const b=[...document.querySelectorAll('button')].filter(x=>x.offsetParent!==null && (x.textContent||'').trim()==='Next').pop(); if(b) b.click();""")
time.sleep(8)
print("step2 URL:", s.url[:130])

# Step 2: attach the ECR read-only managed policy
s.script("""
const doc = arguments[0];
function setValue(el,val){const st=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;st.call(el,val);
 el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}
const searches=[...document.querySelectorAll('input[type=search]')].filter(e=>e.offsetParent!==null);
searches.forEach(e=>{if(e.value) setValue(e,'');});
setValue(searches[searches.length-1], doc);
""", [POLICY])
time.sleep(7)
picked = s.script("""
const name=arguments[0];
const row=[...document.querySelectorAll('tr')].find(r=>(r.textContent||'').includes(name));
if(!row) return 'row not found';
const cb=row.querySelector('input[type=checkbox]');
if(!cb) return 'no checkbox';
if(!cb.checked){cb.scrollIntoView({block:'center'}); cb.click();}
return 'checked=' + cb.checked;
""", [POLICY])
print("policy:", picked)
time.sleep(2)
s.script("""const b=[...document.querySelectorAll('button')].filter(x=>x.offsetParent!==null && (x.textContent||'').trim()==='Next').pop(); if(b) b.click();""")
time.sleep(8)
print("step3 URL:", s.url[:130])

# Step 3: name and create
named = s.script("""
const nm=arguments[0];
const st=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
const inp=[...document.querySelectorAll('input[type=text]')].filter(e=>e.offsetParent!==null)[0];
if(!inp) return 'no name input';
inp.focus(); st.call(inp,nm);
inp.dispatchEvent(new Event('input',{bubbles:true})); inp.dispatchEvent(new Event('change',{bubbles:true}));
return inp.value;
""", [ROLE])
print("role name:", named)
time.sleep(2)
s.script("""const b=[...document.querySelectorAll('button')].filter(x=>x.offsetParent!==null && /^Create role$/.test((x.textContent||'').trim())).pop(); if(b) b.click();""")
time.sleep(12)
print("final URL:", s.url[:130])
print("text:", s.text()[:300].replace("\n"," | "))
