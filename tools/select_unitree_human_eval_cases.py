"""Select a fixed manifest prefix per stratum, preserving completed episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-stratum', type=int, default=4)
    args = parser.parse_args()
    assert args.per_stratum > 0
    source_manifest = args.source/'matched_manifest.json'
    manifest = json.loads(source_manifest.read_text())
    plan = json.loads((args.source/'plan.json').read_text())
    cases = []
    for stratum in ['unfiltered', 'blocked_detour']:
        selected = [c for c in manifest['cases'] if c['stratum'] == stratum][:args.per_stratum]
        assert len(selected) == args.per_stratum
        cases.extend(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'plan.json', dict(plan, count=len(cases)))
    write(args.output/'matched_manifest.json', dict(manifest, cases=cases))
    new_hash = digest(args.output/'matched_manifest.json')
    provenance = dict(source=str(args.source.resolve()), source_manifest_sha256=digest(source_manifest),
                      selection='First cases in original manifest order per stratum; no outcome filtering',
                      selected_seeds=[c['seed'] for c in cases], reused={})
    for name in list(plan['checkpoints'])+['direct_goal']:
        source = args.source/name/'results.json'
        if not source.exists():
            continue
        saved = json.loads(source.read_text())
        assert saved['manifest_sha256'] == digest(source_manifest)
        by_seed = {r['seed']: r for r in saved['episodes']}
        episodes = [by_seed[c['seed']] for c in cases if c['seed'] in by_seed]
        destination = args.output/name
        destination.mkdir()
        write(destination/'results.json', dict(saved, episodes=episodes, manifest_sha256=new_hash))
        for filename in ['contact_geometry.json', 'contact_probe.json']:
            if (source.parent/filename).exists():
                shutil.copy2(source.parent/filename, destination/filename)
        provenance['reused'][name] = dict(source_results_sha256=digest(source),
                                        reused_seeds=[r['seed'] for r in episodes],
                                        completed_source_cases=len(saved['episodes']))
        assert json.loads((destination/'results.json').read_text())['episodes'] == episodes
    if (args.source/'input_audit.json').exists():
        shutil.copy2(args.source/'input_audit.json', args.output/'input_audit.json')
    write(args.output/'selection_provenance.json', provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
