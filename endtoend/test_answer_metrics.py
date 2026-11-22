import ast
from pathlib import Path
import re
import string
import unittest
from .answer_metrics import answers_match, endpoint_answer_metrics, first_trigger_metrics


class AnswerMetricsTests(unittest.TestCase):
    def test_official_function_parity(self):
        source=Path(__file__).resolve().parents[1]/'StreamReady/lavis/tasks/readiness_vqa.py'
        tree=ast.parse(source.read_text())
        names={'normalize_answer','_first_number','answers_match'}
        namespace={'re':re,'_ARTICLES':{'a','an','the'},'_PUNC':str.maketrans('','',string.punctuation)}
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual(len(functions),3)
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
        texts=['','The CAT!','cat','A cat is visible.','3','3 times','three','Yes','No',
               'The person places the plate on the table','put plate on table','13','-2.5']
        for pred in texts:
            for gold in texts:
                for task in ['REC','current_observation']:
                    self.assertEqual(answers_match(pred,gold,task),namespace['answers_match'](pred,gold,task))

    def test_no_answer_and_premature_answer_are_not_hidden(self):
        rows=[dict(task_id='a',current_time=0,probability=.8,expected_ready=0,answer='A CAT!',reference='',qa_type='current_observation'),
              dict(task_id='a',current_time=1,probability=.9,expected_ready=1,answer='cat',reference='cat',qa_type='current_observation'),
              dict(task_id='b',current_time=1,probability=.1,expected_ready=1,answer='3 times',reference='3',qa_type='temporal_comparison'),
              dict(task_id='c',current_time=0,probability=.1,expected_ready=0,answer='anything',reference='',qa_type='temporal_comparison')]
        endpoint=endpoint_answer_metrics(rows,.5)
        self.assertEqual(endpoint['answer_accuracy'],1)
        self.assertEqual(endpoint['gated_answer_accuracy'],.5)
        first=first_trigger_metrics(rows,.5)
        self.assertEqual(first['answer_accuracy'],.5)
        self.assertEqual(first['on_time_and_text_correct'],0)
        self.assertEqual(first['early_tasks'],1)
        self.assertEqual(first['no_answer_tasks'],1)
        self.assertEqual(first['unanswerable_tasks'],1)

    def test_empty_subset_and_numeric_matching(self):
        self.assertIsNone(endpoint_answer_metrics([],.5)['answer_accuracy'])
        self.assertTrue(answers_match('3 times','3','REC'))
        self.assertFalse(answers_match('three','3','REC'))  # Official task matcher is numeric regex only.
        self.assertFalse(answers_match('No','Yes'))


if __name__ == '__main__':
    unittest.main()
