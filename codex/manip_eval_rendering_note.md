# Manip Eval Rendering Note

- Context: `eval_interactive_manip.py` for cube/manip intervention debugging.
- Problem: custom `pygame` RGB viewer caused flicker and window instability; manip env does not provide Gym-native `human` rendering path.
- Fix: use native passive MuJoCo viewer from manip env (`launch_passive_viewer`, `sync_passive_viewer`, `close_passive_viewer`), and keep the keyboard control window independent.
