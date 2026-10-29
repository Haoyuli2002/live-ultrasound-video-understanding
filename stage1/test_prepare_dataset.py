import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.asr_pipeline import transcribe_audio, transcribe_video
from stage1.alignment import segment_words
from stage1.prepare_dataset import Policy, build, paired_rows, prepare_video
from stage1.restore_punctuation import timed_words, restore_transcript
from stage1.transcribe import run


def segment(text, start=0, seconds=3):
    tokens=text.split()
    return {'start':start,'end':start+seconds,'text':text,
            'avg_logprob':-.2,'compression_ratio':1.1,'no_speech_prob':.01,
            'words':[{'word':word,'start':start+i*seconds/len(tokens),
                      'end':start+(i+1)*seconds/len(tokens),'probability':.95}
                     for i,word in enumerate(tokens)]}


def transcript():
    return {'video_id':'v','duration_sec':30,'segments':[
        segment('The probe shows the liver.',0),
        segment('The kidney appears below it.',3),
        segment('Move the probe slightly lower.',6)]}


class DatasetBuildTests(unittest.TestCase):
    def test_word_times_survive_sentence_split_and_triplets(self):
        t=transcript()
        # Deliberately nonuniform word timing: character interpolation would differ.
        t['segments'][1]['words'][0]['end']=3.1
        t['segments'][1]['words'][1]['start']=3.1
        units,review,audit=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        self.assertEqual(review,[])
        self.assertEqual([u['text'] for u in units],[s['text'] for s in t['segments']])
        self.assertEqual({u['alignment'] for u in units},{'whisper_word_timestamps'})
        rows=list(paired_rows('v',units,'hash'))
        self.assertEqual(len(rows),6)
        before,through,masked=rows[:3]
        self.assertEqual(before['video_window'],[0,3])
        self.assertEqual(through['video_window'],[0,6])
        self.assertEqual(masked['historical_asr'],'')
        self.assertEqual(before['historical_asr'],t['segments'][0]['text'])
        self.assertEqual({r['target'] for r in rows[:3]},{t['segments'][1]['text']})

    def test_strict_alignment_never_silently_estimates(self):
        s=segment('The kidney is visible.')
        s.pop('words')
        with self.assertRaisesRegex(ValueError,'missing_word_timestamps'): segment_words(s)
        self.assertEqual(segment_words(s,allow_estimated=True)[0]['alignment'],
                         'segment_character_interpolation')
        s=segment('The kidney is visible.')
        s['words'][0]['end']=20
        with self.assertRaisesRegex(ValueError,'invalid_word_time'):
            segment_words(s,allow_estimated=True)

    def test_changed_medical_word_requires_realignment(self):
        s=segment('The kidney is visible.')
        s['text']='The liver is visible.'
        with self.assertRaisesRegex(ValueError,'word_text_mismatch'):
            segment_words(s)

    def test_native_compound_pieces_and_zero_duration_words_are_not_rewritten(self):
        s={'start':0,'end':5,'text':'A four-chamber view is visible.',
           'words':[{'word':text,'start':start,'end':end,'probability':.9}
                    for text,start,end in [(' A',0,0),(' four',0,1),('-chamber',1,2),
                                           (' view',2,3),(' is',3,4),(' visible.',4,5)]]}
        units,_,_=prepare_video({'segments':[s]},Policy(min_words=3,min_seconds=.5))
        self.assertEqual(units[0]['text'],s['text'])
        self.assertEqual(units[0]['zero_duration_word_count'],1)

    def test_quoted_sentence_and_unrounded_target_duration(self):
        t={'segments':[segment('The probe is here.',0,3),
                       segment('One two three four five six seven eight nine ten."',3,4.9999)]}
        units,_,_=prepare_video(t,Policy())
        self.assertEqual(units[-1]['text'],t['segments'][-1]['text'])
        self.assertFalse(units[-1]['target_eligible'])
        self.assertEqual(list(paired_rows('v',units,'h')),[])

    def test_llm_restoration_uses_native_word_times_and_cannot_rewrite(self):
        class Boundaries:
            def boundaries(self,words,start,end,context):
                return [{'after':i,'mark':'.'} for i,w in enumerate(words)
                        if w['text'].endswith('.') and start<=i<end]
        t=transcript()
        actual=timed_words(t['segments'])
        self.assertEqual(actual[1]['start'],t['segments'][0]['words'][1]['start'])
        restored,audit=restore_transcript(t,Boundaries())
        self.assertEqual(audit['alignment'],'whisper_word_timestamps')
        self.assertEqual(restored['sentence_units'][1]['start'],3)

    def test_quarantined_text_never_enters_history_and_gap_not_joined(self):
        t=transcript()
        t['segments'].insert(1,segment('This is unfinished',3,1))
        bad=segment('bad bad bad bad bad bad bad bad',4,1)
        bad['avg_logprob']=-4
        t['segments'].insert(2,bad)
        t['segments'][3]=segment('The kidney appears below it.',6)
        t['segments'][4]=segment('Move the probe slightly lower.',9)
        units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        text=' '.join(u['text'] for u in units)
        self.assertNotIn('unfinished',text)
        self.assertNotIn('bad',text)
        self.assertTrue(any(r['scope']=='unfinished_tail' for r in review))
        for row in paired_rows('v',units,'h'):
            self.assertNotIn('bad',row['historical_asr'])

    def test_repetition_and_impossible_time_are_quarantined(self):
        t=transcript()
        t['segments'] += [segment('And long long long long long long long long.',9,3),
                           segment('You want to be in that situation.',12,.007)]
        units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        self.assertEqual(len(units),3)
        reasons={r for item in review for r in item['reasons']}
        self.assertIn('suspected_repetition_loop',reasons)
        self.assertIn('speech_rate',reasons)
        self.assertIn('sentence_duration',reasons)

    def test_duplicate_sentences_across_segments_all_flagged(self):
        t=transcript()
        t['segments']=[segment('The kidney is visible.',i*3) for i in range(3)]
        units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        self.assertEqual(units,[])
        self.assertEqual(sum('consecutive_duplicate_sentences' in x['reasons'] for x in review),3)

    def test_unfinished_tail_and_abbreviation_decimal_not_forced(self):
        t=transcript()
        t['segments']=[segment('Dr. Smith measures 1.5 cm in this view.',0,5),
                       segment('Then the probe',5,2)]
        units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        self.assertEqual(len(units),1)
        self.assertEqual(units[0]['text'],'Dr. Smith measures 1.5 cm in this view.')
        self.assertEqual(review[0]['scope'],'unfinished_tail')

    def test_target_minimums_apply_to_whole_triplet_but_preserve_history(self):
        short='The probe shows the liver.'
        words='The probe is now positioned over the right upper abdominal region.'
        t={'video_id':'v','duration_sec':30,'segments':[
            segment(short,0,3),
            segment('This view shows a kidney.',3,5),
            segment(words,8,4.999),
            segment(words.rstrip('.')+' while tilting slightly.',13,5)]}
        units,review,_=prepare_video(t,Policy())
        rows=list(paired_rows('v',units,'h'))
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['sentence_window']==[13,18] for r in rows))
        self.assertTrue(all(len(r['target'].split())>=10 for r in rows))
        self.assertIn(short,rows[0]['historical_asr'])
        self.assertTrue(any(r.get('action')=='exclude_target_keep_history' for r in review))
        # Both thresholds are inclusive: exactly 5 seconds and exactly 10 words survive.
        exact='One two three four five six seven eight nine ten.'
        t['segments']=[segment(short,0,3),segment(exact,3,5)]
        units,_,_=prepare_video(t,Policy())
        self.assertEqual(len(list(paired_rows('v',units,'h'))),3)

    def test_long_informative_narration_has_no_length_ceiling(self):
        long_text=('Starting with the apical four chamber view the operator tilts the probe '
                   'superiorly to bring the left ventricular outflow tract into view while '
                   'comparing right and left ventricular size and observing the atria and '
                   'the mitral and tricuspid valves throughout the cardiac cycle; '
                   'the next part demonstrates how small changes in probe angle affect '
                   'the displayed anatomy and how the image returns to the initial view '
                   'when the operator reverses the tilt without moving the contact point '
                   'away from the apex of the heart.')
        self.assertGreater(len(long_text.split()),80)
        t={'segments':[segment('The probe is near the apex.',0,3),
                       segment(long_text,3,70)]}
        units,review,_=prepare_video(t,Policy())
        rows=list(paired_rows('v',units,'h'))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['target'],long_text)
        self.assertEqual(rows[0]['sentence_window'],[3,73])
        self.assertFalse(any(x['scope']=='sentence' for x in review))

    def test_source_confidence_and_video_bounds_checked(self):
        for key,value,reason in [('avg_logprob',-2,'low_asr_logprob'),
                                 ('compression_ratio',4,'high_asr_compression')]:
            t=transcript(); t['segments'][1][key]=value
            units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
            self.assertNotIn('The kidney appears below it.',[u['text'] for u in units])
            self.assertIn(reason,review[0]['reasons'])
        t=transcript(); t['duration_sec']=4
        units,review,_=prepare_video(t,Policy(min_words=3,min_seconds=.5))
        self.assertTrue(any('asr_exceeds_video_duration' in r['reasons'] for r in review))

    def test_manifest_coverage_and_rejected_dataset_cannot_be_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'raw';src.mkdir()
            original=json.dumps(transcript())
            (src/'v.json').write_text(original)
            mapping=root/'map.json'; mapping.write_text(json.dumps({'v':'v.mp4'}))
            out=root/'v2'
            report=build(src,mapping,out,Policy(min_words=3,min_seconds=.5))
            self.assertEqual(report['status'],'ready')
            self.assertEqual(report['counts']['samples'],6)
            self.assertEqual((src/'v.json').read_text(),original)
            self.assertIn('v',json.loads((out/'manifest.json').read_text())['sources'])
            with self.assertRaises(FileExistsError): build(src,mapping,out,Policy(min_words=3,min_seconds=.5))
            mapping.write_text(json.dumps({'v':'v.mp4','missing':'missing.mp4'}))
            report=build(src,mapping,root/'failed',Policy(min_words=3,min_seconds=.5))
            self.assertEqual(report['status'],'rejected')
            self.assertFalse((root/'failed/samples.jsonl').exists())
            self.assertEqual(report['failed_videos'],['missing'])


class ASRTests(unittest.TestCase):
    def test_decoder_options_and_word_confidence_are_saved(self):
        word=SimpleNamespace(start=0.123456,end=1.4,word=' Liver.',probability=.83)
        seg=SimpleNamespace(start=0,end=2,text=' Liver.',words=[word],
                            avg_logprob=-.3,compression_ratio=1.2,no_speech_prob=.01,temperature=0)
        info=SimpleNamespace(language='en',language_probability=.99,duration=2)
        class Fake:
            def transcribe(self,audio,**kwargs):
                self.options=kwargs
                return iter([seg]),info
        model=Fake()
        with patch('scripts.asr_pipeline.importlib.metadata.version',return_value='test'):
            result=transcribe_audio('audio.wav',model=model)
        self.assertTrue(model.options['word_timestamps'])
        self.assertFalse(model.options['condition_on_previous_text'])
        self.assertEqual(result['segments'][0]['words'][0]['start'],.123456)
        self.assertEqual(result['segments'][0]['avg_logprob'],-.3)

    def test_audio_removed_on_recognition_error(self):
        paths=[]
        def extract(video,path):
            Path(path).write_bytes(b'audio');paths.append(Path(path));return path
        with patch('scripts.asr_pipeline.extract_audio',side_effect=extract), \
             patch('scripts.asr_pipeline.transcribe_audio',side_effect=RuntimeError('decode')):
            with self.assertRaisesRegex(RuntimeError,'decode'):
                transcribe_video('v.mp4')
        self.assertTrue(paths)
        self.assertFalse(paths[0].exists())

    def test_asr_resume_checks_video_contents_and_does_not_reload_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); video=root/'v.mp4'; video.write_bytes(b'video')
            mapping=root/'map.json';mapping.write_text(json.dumps({'v':str(video)}))
            out=root/'asr'
            with patch('stage1.transcribe.importlib.metadata.version',return_value='test'), \
                 patch('stage1.transcribe.asr_pipeline.load_whisper_model') as load, \
                 patch('stage1.transcribe.asr_pipeline.transcribe_video',return_value=transcript()) as recognize:
                self.assertEqual(run(mapping,out)['failed'],[])
                self.assertEqual(load.call_count,1)
                result=run(mapping,out,resume=True)
                self.assertEqual(result['skipped_identical'],['v'])
                self.assertEqual(load.call_count,1)
                video.write_bytes(b'changed')
                self.assertEqual(run(mapping,out,resume=True)['failed'],['v'])
                self.assertEqual(recognize.call_count,1)

    def test_cluster_relative_video_map_uses_explicit_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'cluster_data/videos').mkdir(parents=True)
            video=root/'cluster_data/videos/v.mp4'; video.write_bytes(b'video')
            maps=root/'cluster_data/splits'; maps.mkdir()
            mapping=maps/'map.json'; mapping.write_text(json.dumps({'v':'cluster_data/videos/v.mp4'}))
            with patch('stage1.transcribe.importlib.metadata.version',return_value='test'), \
                 patch('stage1.transcribe.asr_pipeline.load_whisper_model'), \
                 patch('stage1.transcribe.asr_pipeline.transcribe_video',return_value=transcript()) as recognize:
                report=run(mapping,root/'asr',video_root=root)
                self.assertEqual(report['failed'],[])
                self.assertEqual(recognize.call_args.args[0],video.resolve())
                saved=json.loads((root/'asr/selected_video_map.json').read_text())
                self.assertEqual(saved,{'v':str(video.resolve())})


if __name__=='__main__':
    unittest.main()
