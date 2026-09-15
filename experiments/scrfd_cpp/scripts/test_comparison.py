"""Check that comparison does not assume detector track IDs correspond."""
import csv
from pathlib import Path
import tempfile
import unittest
from benchmark import compare

FIELDS=['frame','time_s','track','score','speaking','x1','y1','x2','y2']

class ComparisonTest(unittest.TestCase):
    def compare_rows(self, baseline, candidate):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ['baseline.csv','candidate.csv']]
            for path,rows in zip(paths,[baseline,candidate]):
                with path.open('w') as f:
                    writer=csv.writer(f);writer.writerow(FIELDS);writer.writerows(rows)
            return compare(*paths)

    def test_reordered_ids_and_missing_boxes(self):
        base=[[0,0,0,1,1,0,0,10,10],[0,0,1,-1,0,20,0,30,10],
              [1,.04,0,1,1,0,0,10,10],[1,.04,1,-1,0,20,0,30,10]]
        candidate=[[0,0,9,-1,0,20,0,30,10],[0,0,8,1,1,0,0,10,10],
                   [1,.04,10,-1,0,0,0,10,10],[1,.04,11,1,1,50,0,60,10]]
        result=self.compare_rows(base,candidate)
        self.assertEqual(result['matched'],3)
        self.assertEqual(result['unmatched_baseline'],1)
        self.assertEqual(result['unmatched_candidate'],1)
        self.assertAlmostEqual(result['speaking_agreement_pct'],200/3)
        self.assertEqual(result['candidate_track_changes_per_baseline_track'],1)

    def test_no_matches_is_not_perfect_agreement(self):
        result=self.compare_rows([[0,0,0,1,1,0,0,10,10]],[])
        self.assertEqual(result['matched'],0)
        self.assertIsNone(result['speaking_agreement_pct'])

if __name__=='__main__':unittest.main()
