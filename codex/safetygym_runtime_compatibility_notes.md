# SafetyGym Runtime Compatibility Notes

- Environment is offline; direct `pip install xmltodict` is not possible.
- Added a local `xmltodict.py` shim (`parse`/`unparse` subset) so Safety-Gymnasium can parse/build MuJoCo XML without external dependency download.
- Patched `safety-gymnasium/safety_gymnasium/utils/registration.py` for newer Gymnasium `EnvSpec` APIs that no longer expose `apply_api_compatibility` and `autoreset`.
- Updated SafetyGym helper scripts to use `conda run -n <env>` instead of `conda activate` so they work in non-interactive shells.
