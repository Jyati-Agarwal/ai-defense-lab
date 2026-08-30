#!/usr/bin/env python3
"""Attach an AWS managed policy to an IAM user through the console UI.

Reuses the console's "attach policies directly" wizard, which propagates to
React state reliably (unlike programmatic edits to the JSON policy editor).
"""
import sys
import time

sys.path.insert(0, "scripts")
from ffsession import Session

USER = sys.argv[1]
POLICY = sys.argv[2]
s = Session()

s.goto(f"https://us-east-1.console.aws.amazon.com/iam/home#/users/details/{USER}/add-permissions", settle=18)
print("radio:", (s.click_radio("Attach policies directly") or "")[:45])
time.sleep(5)

typed = s.script("""
const doc = arguments[0];
function setValue(el, val){
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
  setter.call(el, val);
  el.dispatchEvent(new Event('input',{bubbles:true}));
  el.dispatchEvent(new Event('change',{bubbles:true}));
}
const searches=[...document.querySelectorAll('input[type=search]')].filter(e=>e.offsetParent!==null);
searches.forEach(e=>{ if(e.value) setValue(e,''); });
const filter = searches[searches.length-1];
filter.focus(); setValue(filter, doc);
return searches.length;
""", [POLICY])
print("filter boxes:", typed)
time.sleep(7)

picked = s.script("""
const name = arguments[0];
const row=[...document.querySelectorAll('tr')].find(r=>(r.textContent||'').includes(name));
if(!row) return 'row not found';
const cb=row.querySelector('input[type=checkbox]');
if(!cb) return 'no checkbox';
if(!cb.checked){ cb.scrollIntoView({block:'center'}); cb.click(); }
return 'checked=' + cb.checked;
""", [POLICY])
print("select:", picked)
if "checked=true" not in str(picked):
    sys.exit(f"could not select {POLICY}")

time.sleep(2)
s.script("""
const b=[...document.querySelectorAll('button')].filter(x=>x.offsetParent!==null && (x.textContent||'').trim()==='Next').pop();
if(b){b.scrollIntoView({block:'center'}); b.click();}
""")
time.sleep(7)
s.script("""
const b=[...document.querySelectorAll('button')].filter(x=>x.offsetParent!==null && (x.textContent||'').trim()==='Add permissions').pop();
if(b){b.scrollIntoView({block:'center'}); b.click();}
""")
time.sleep(10)
print("final URL:", s.url[:130])
print("attached:", POLICY, "->", USER)
