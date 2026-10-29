"""Resumable, map-selected word ASR. Existing outputs require identical provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from scripts import asr_pipeline
from .restore_punctuation import write_json_atomic


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def recipe(model: str, device: str, language: str) -> dict:
    return {'schema':'stage1_word_asr_v2', 'model':model, 'device':device,
            'language':language, 'word_timestamps':True, 'condition_on_previous_text':False,
            'asr_implementation_sha256':file_hash(Path(asr_pipeline.__file__)),
            'wrapper_sha256':file_hash(Path(__file__)),
            'faster_whisper_version':importlib.metadata.version('faster-whisper')}


def resume_matches(saved: dict, request: dict) -> bool:
    return saved.get('build_request') == request


def run(video_map: Path, output_dir: Path, *, model_name='large-v3', device='cuda',
        language='en', resume=False, limit_videos=None) -> dict:
    selected = json.loads(video_map.read_text())
    if not isinstance(selected,dict) or not selected: raise ValueError('Expected nonempty video map')
    if limit_videos is not None and limit_videos < 1: raise ValueError('limit_videos must be positive')
    if any(not isinstance(k,str) or not k or Path(k).name!=k or k in {'.','..'} for k in selected):
        raise ValueError('Unsafe video ID')
    selected = dict(sorted(selected.items())[:limit_videos])
    resolved = {k:(Path(v) if Path(v).is_absolute() else (video_map.parent/v).resolve())
                for k,v in selected.items()}
    if any(not p.is_file() or p.stem!=k for k,p in resolved.items()):
        raise ValueError('Every mapped video must exist and its stem must equal the video ID')
    settings = recipe(model_name,device,language)
    if output_dir.exists() and not resume: raise FileExistsError(f'Use a new output directory: {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    previous = output_dir/'run_config.json'
    config={'recipe':settings,'selected_video_map':{k:str(v) for k,v in resolved.items()}}
    if previous.exists() and json.loads(previous.read_text())!=config:
        raise ValueError('ASR resume configuration differs; use a new output directory')
    write_json_atomic(previous,config)
    write_json_atomic(output_dir/'selected_video_map.json',config['selected_video_map'])
    model, failures, completed, skipped = None, [], [], []
    for video_id,video in resolved.items():
        destination=output_dir/f'{video_id}.json'
        try:
            request={'recipe':settings,'video_sha256':file_hash(video),'video_path':str(video.resolve())}
            if destination.exists():
                if not resume or not resume_matches(json.loads(destination.read_text()),request):
                    raise ValueError('Existing transcript provenance differs; use a new output directory')
                skipped.append(video_id)
                continue
            if model is None: model=asr_pipeline.load_whisper_model(model_name,device)
            transcript=asr_pipeline.transcribe_video(video,model_size=model_name,device=device,
                                                    language=language,model=model)
            if transcript is None: raise RuntimeError('Audio extraction failed')
            transcript['build_request']=request
            write_json_atomic(destination,transcript)
            completed.append(video_id)
            audit={'video_id':video_id,'status':'transcribed','video_sha256':request['video_sha256']}
        except Exception as error:
            failures.append(video_id)
            audit={'video_id':video_id,'status':'failed','error':str(error)}
        with (output_dir/'asr_audit.jsonl').open('a') as stream:
            stream.write(json.dumps(audit,ensure_ascii=False)+'\n')
        print(json.dumps(audit,ensure_ascii=False),flush=True)
    report={'selected':len(selected),'completed':completed,'skipped_identical':skipped,'failed':failures}
    write_json_atomic(output_dir/'asr_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-map',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',default='large-v3')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--language',default='en')
    p.add_argument('--limit-videos',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    report=run(args.video_map,args.output_dir,model_name=args.model,device=args.device,
               language=args.language,resume=args.resume,limit_videos=args.limit_videos)
    print(json.dumps(report,ensure_ascii=False))
    if report['failed']: raise SystemExit(1)


if __name__=='__main__':
    main()
