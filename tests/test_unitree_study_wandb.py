import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'exporter', Path(__file__).resolve().parents[1]/'scripts/cluster/upload_unitree_study_wandb.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_stable_host_qualified_id(self):
        self.assertEqual(exporter.run_id('laptop', 'alpha73'), exporter.run_id('laptop', 'alpha73'))
        self.assertNotEqual(exporter.run_id('desktop', 'alpha73'), exporter.run_id('laptop', 'alpha73'))

    def test_credentials_and_paths_excluded(self):
        clean = exporter.clean({'api_key':'private', 'nested':{'password':'private'},
                                'model':'/home/user/checkpoint.pt', 'absent':float('inf')})
        self.assertNotIn('api_key', clean)
        self.assertEqual(clean['nested'], {})
        self.assertEqual(clean['model'], 'local-file:checkpoint.pt')
        self.assertEqual(clean['absent'], 'nonfinite:inf')

    def test_live_jsonl_ignores_only_unterminated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'metrics.jsonl'
            p.write_text('{"step":100}\n{"step":2')
            self.assertEqual(exporter.jsonl(p), [{'step':100}])
            p.write_text('broken\n')
            with self.assertRaises(json.JSONDecodeError): exporter.jsonl(p)

    def test_table_nonfinite_diagnostics_preserved_without_mixed_column(self):
        columns, rows = exporter.table_data([{'distance':2.0}, {'distance':'nonfinite:inf'}])
        self.assertEqual(rows[0][columns.index('distance')], 2.0)
        self.assertIsNone(rows[1][columns.index('distance')])
        self.assertEqual(json.loads(rows[1][-1]), {'distance':'nonfinite:inf'})

    def test_interrupted_upload_cursor_recovers_without_duplicate_history(self):
        rows = [{'step':100}, {'step':200}, {'step':300}]
        self.assertEqual(exporter.history_cursor({'train/decision':200}, rows, 'train'), 2)
        self.assertEqual(exporter.history_cursor({'import/training_rows':1,
            'import/training_prefix_sha256':exporter.digest(rows[:1]), 'train/decision':200}, rows, 'train'), 2)
        with self.assertRaises(AssertionError):
            exporter.history_cursor({'train/decision':250}, rows, 'train')

    def test_cost_frontier_is_not_success_score(self):
        rows = []
        for name, host, kind, cost, success in [
            ('low', 'laptop', 'policy', 1, .8), ('high', 'laptop', 'policy', 10, 1),
            ('teacher', 'laptop', 'baseline', 0, 1), ('other', 'desktop', 'policy', 20, 1)]:
            rows.append(dict(name=name, host=host, kind=kind, cohort='validation',
                             collision_episode_rate=cost/100, collision_cost_sum=cost,
                             max_episode_collision_cost=cost, success_rate=success))
        result = {r['name']:r['cost_frontier'] for r in exporter.frontier(rows)}
        self.assertEqual(result, {'low':True, 'high':False, 'teacher':False, 'other':True})

    def test_intervention_average_covers_windows(self):
        rows = [dict(step=100, teacher_fraction_cumulative=1, teacher_fraction_interval=1),
                dict(step=200, teacher_fraction_cumulative=.5, teacher_fraction_interval=0)]
        result = exporter.intervention_summary(rows)
        self.assertEqual(result['teacher_interval_time_average'], .5)
        self.assertEqual(result['final_teacher_fraction'], .5)

    def test_all_cases_retained_and_cost_decomposition_checked(self):
        case = dict(root_pose_relative=[0,0,0,1,0,0,0], goal_xy_relative=[1,0], tile_index=0)
        episodes = [dict(manifest_episode_index=i, start_local_xy=[0,0], goal_local_xy=[1,0],
                         start_quaternion=[1,0,0,0], terrain_tile_id=0,
                         cost_sum=0, collision_cost_sum=0, fall_cost_sum=0) for i in range(64)]
        manifest = {'episodes':[case]*64}
        exporter.check_episodes({'episodes':episodes}, manifest)
        episodes[63]['cost_sum'] = 1
        with self.assertRaises(AssertionError): exporter.check_episodes({'episodes':episodes}, manifest)
        with self.assertRaises(AssertionError): exporter.check_episodes({'episodes':episodes[:63]}, manifest)


if __name__ == '__main__':
    unittest.main()
