"""Browser acceptance checks for the video QA explorer (requires Playwright)."""
import argparse,json,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8767/viz/qa-examples/');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
base=args.url;data=json.load(urllib.request.urlopen(base+'data.json'))
report={'videos':0,'qa':0,'errors':[],'range_checks':0,'playback_checks':0}
with sync_playwright() as pw:
 browser=pw.chromium.launch(args=['--no-sandbox']);page=browser.new_page(viewport={'width':1440,'height':1100})
 page.on('pageerror',lambda e:report['errors'].append(str(e)))
 page.goto(base);expect(page.locator('#explorer')).to_be_visible()
 def at(t):page.wait_for_function('(t)=>Math.abs(document.querySelector("#player").currentTime-t)<0.12 && !document.querySelector("#player").seeking',arg=t,timeout=30000)
 for v in data['videos']:
  page.locator(f'[data-video="{v["id"]}"]').click();page.wait_for_function('document.querySelector("#player").readyState>=2',timeout=30000)
  expect(page.locator('#media-error')).to_be_hidden()
  for t in v['tasks']:
   page.locator(f'[data-task="{t["task_id"]}"]').click();expect(page.locator('#question')).to_have_text(t['question']);expect(page.locator('#answer')).to_have_text(t['reference_answer']);at(t['query_time']);report['qa']+=1
  delayed=next(t for t in v['tasks'] if t['qa_type']=='event_notification')
  page.locator(f'[data-task="{delayed["task_id"]}"]').click();at(delayed['query_time']);expect(page.locator('#phase')).to_contain_text('WAIT')
  page.locator('#go-answer').click();at(delayed['answer_time']);expect(page.locator('#phase')).to_contain_text('ANSWER')
  page.locator('#go-query').click();at(delayed['query_time'])
  page.evaluate('document.querySelector("#player").playbackRate=16');page.locator('#play-window').click()
  page.wait_for_function('(a)=>{const v=document.querySelector("#player");return v.paused && Math.abs(v.currentTime-a)<0.12}',arg=delayed['answer_time'],timeout=30000);report['playback_checks']+=1
  recall=next(t for t in v['tasks'] if t['qa_type']=='history_recall');page.locator(f'[data-task="{recall["task_id"]}"]').click();page.locator('#evidence-details summary').click();page.locator('#evidence button').first.click();at(recall['evidence'][0]['time'])
  for kind in data['counts']['types']:
   page.select_option('#type-filter',kind);expect(page.locator('.qa-option')).to_have_count(sum(t['qa_type']==kind for t in v['tasks']))
  page.select_option('#type-filter','');page.fill('#search','__no_such_question__');expect(page.locator('.qa-option')).to_have_count(0);expect(page.locator('.empty')).to_be_visible();page.fill('#search','')
  url=urllib.request.Request(urllib.parse.urljoin(base,v['url']),headers={'Range':'bytes=0-1023'})
  with urllib.request.urlopen(url) as response:assert response.status==206 and len(response.read())==1024;report['range_checks']+=1
  report['videos']+=1;print(v['id'],'passed',flush=True)
 # Long waiting interval, bookmarking and instant context.
 v=data['videos'][-1];t=max(v['tasks'],key=lambda t:t['answer_time']-t['query_time']);page.locator(f'[data-task="{t["task_id"]}"]').click();link=page.url;page.reload();expect(page.locator('#question')).to_have_text(t['question']);at(t['query_time'])
 page.evaluate('document.querySelector("#player").playbackRate=16');page.locator('#play-window').click();page.wait_for_function('(a)=>{const v=document.querySelector("#player");return v.paused&&Math.abs(v.currentTime-a)<.12}',arg=t['answer_time'],timeout=30000);report['long_interval_seconds']=t['answer_time']-t['query_time']
 page.evaluate('document.fonts.ready');page.screenshot(path=str(args.output/'desktop.png'),full_page=True)
 immediate=next(t for t in v['tasks'] if t['qa_type']=='current_observation');page.locator(f'[data-task="{immediate["task_id"]}"]').click();page.locator('#play-window').click();page.wait_for_function('(a)=>{const v=document.querySelector("#player");return v.paused&&Math.abs(v.currentTime-a)<.12}',arg=immediate['answer_time']+3,timeout=10000)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth');page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
 # Rapidly changing videos must leave the final question and seek in sync.
 page.evaluate('()=>{const bs=document.querySelectorAll("[data-video]");bs[0].click();bs[1].click();bs[2].click();}')
 v=data['videos'][2];chosen=next(t for t in v['tasks'] if t['task_id']==v['featured_task']);expect(page.locator('#question')).to_have_text(chosen['question']);at(chosen['query_time'])
 browser.close()
assert not report['errors'],report
report['passed']=True;(args.output/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
