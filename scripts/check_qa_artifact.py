#!/usr/bin/env python3
"""Browser acceptance checks for the exported QA preview and local media binding."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/qa_demo_artifact_20261124'


def main():
    source = json.loads((ROOT / 'viz/qa-examples/data.json').read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 820, 'height': 1200})
        errors, external = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        def block_external(route):
            external.append(route.request.url)
            route.abort()
        page.route('http://**/*', block_external)
        page.route('https://**/*', block_external)
        page.goto((OUT / 'preview.html').as_uri())
        page.frame_locator('iframe').locator('[data-field="question"]').wait_for()
        frame = page.frames[1]
        def field(name):
            return frame.locator(f'[data-field="{name}"]')
        assert field('player').is_hidden()
        assert field('go-query').is_disabled()
        checked = frame.evaluate('''async () => {
          const data=JSON.parse(document.getElementById('ultrasound-qa-data').textContent);
          const el=n=>document.querySelector('[data-field="'+n+'"]');
          const choose=(name,value)=>{el(name).value=value;el(name).dispatchEvent(new Event('change'));};
          let qa=0,decisions=0,evidence=0;
          for(const v of data.videos){choose('video',v.id);for(const t of v.tasks){choose('task',t.task_id);
            if(el('question').textContent!==t.question||el('answer').textContent!==t.reference_answer)throw Error(t.task_id);
            for(let i=0;i<t.decisions.length;i++){el('decisions').children[i].click();
              if(!el('selected-decision').textContent.endsWith(t.decisions[i].label))throw Error('decision');decisions++;}
            for(let i=0;i<t.evidence.length;i++){el('evidence-controls').children[i].click();await el('image').decode();evidence++;}
            qa++;
          }}return {qa,decisions,evidence};
        }''')
        field('files').set_input_files({'name':'unmatched.mp4','mimeType':'video/mp4','buffer':b'not a video'})
        assert '未匹配' in field('media-error').inner_text()
        assert field('go-query').is_disabled()
        first = source['videos'][0]
        field('video').select_option(first['id'])
        first_path = (ROOT / 'viz/qa-examples' / first['url']).resolve()
        field('files').set_input_files(str(first_path))
        frame.wait_for_function('document.querySelector("video").readyState>=2')
        field('video').select_option(source['videos'][1]['id'])
        assert field('player').is_hidden() and field('go-query').is_disabled()
        files = [str((ROOT / 'viz/qa-examples' / v['url']).resolve()) for v in source['videos']]
        field('files').set_input_files(files)
        videos_checked = 0
        def at_time(seconds):
            frame.wait_for_function('t=>{const p=document.querySelector("video");return p.readyState>=2&&!p.seeking&&Math.abs(p.currentTime-t)<.1}', arg=seconds)
        for video in source['videos']:
            print('Video',video['id'],flush=True)
            field('video').select_option(video['id'])
            task = next(t for t in video['tasks'] if t['task_id'] == video['featured_task'])
            at_time(task['query_time'])
            field('go-answer').click(); at_time(task['answer_time'])
            field('go-query').click(); at_time(task['query_time'])
            field('evidence-controls').locator('button').first.click(); at_time(task['evidence'][0]['time'])
            field('decisions').locator('button').first.click(); at_time(task['decisions'][0]['time'])
            assert not field('go-query').is_disabled()
            videos_checked += 1
        field('video').select_option(first['id'])
        delayed = next(t for t in first['tasks'] if t['task_id'] == first['featured_task'])
        at_time(delayed['query_time'])
        frame.evaluate('document.querySelector("video").playbackRate=8')
        field('play-window').click()
        frame.wait_for_function('t=>{const p=document.querySelector("video");return p.paused&&Math.abs(p.currentTime-t)<.1}', arg=delayed['answer_time'], timeout=20000)
        instant = next(t for t in first['tasks'] if t['query_time'] == t['answer_time'])
        field('task').select_option(instant['task_id']);at_time(instant['query_time'])
        assert '前后各 3 秒' in field('play-window').inner_text()
        field('play-window').click()
        frame.wait_for_function('t=>{const p=document.querySelector("video");return p.paused&&Math.abs(p.currentTime-t)<.1}', arg=instant['answer_time']+3)
        field('task').select_option(delayed['task_id']);field('play-window').click()
        field('video').select_option(source['videos'][1]['id'])
        second_task=next(t for t in source['videos'][1]['tasks'] if t['task_id']==source['videos'][1]['featured_task'])
        at_time(second_task['query_time']);page.wait_for_timeout(400)
        assert field('player').evaluate('(p)=>p.paused')
        field('video').select_option(first['id']);at_time(delayed['query_time'])
        for width in (820,390,320):
            page.set_viewport_size({'width':width,'height':1400});page.wait_for_timeout(150)
            assert frame.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),width
            page.screenshot(path=str(OUT/f'video-preview-{width}.png'),full_page=True)
        assert not errors, errors
        assert not external, external
        browser.close()
    result={**checked,'videos_playable':videos_checked,'unknown_filename_rejected':True,
            'missing_video_controls_disabled':True,'delayed_interval_autopause':True,
            'instant_context_autopause':True,'switch_cancels_previous_playback':True,
            'widths':[820,390,320],'javascript_errors':errors,'external_requests':external}
    (OUT/'video_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
