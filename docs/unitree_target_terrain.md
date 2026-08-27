# Unitree Target Terrain

`unitree_target_terrain.py` defines the proposed continuous target benchmark.
It replaces isolated terrain cells with one 24 x 24 m MuJoCo arena.

Independent seeded brush layers paint:

- local rigid, blue low-friction ice, and yellow high-friction sand surfaces;
- rough ground, rubble, stairs, and ramps;
- red static barriers and magenta periodic moving obstacles.

Geometry and material brushes can overlap. Geometry remains compositional, so
rough ice and sandy stairs are valid. Material paint is exclusive and follows
last-paint-wins semantics: the final 0.5 m raster contains exactly one physical
surface per location. The visible material therefore owns the contact friction
and no hidden ice/sand geom can contribute a second contact.

Start and goal are non-colliding markers on the same terrain, not separate
platforms. Every preview exports the exact layout metadata used to compile the
MuJoCo model.

## Calibrated sand proxy

MuJoCo rigid contact friction changes traction rather than providing a useful
viscous sand slowdown. High friction and deep contact compliance destabilized
the flat locomotion policy; contact margin had no useful speed effect. The
practical benchmark proxy is therefore a horizontal resistance force applied
only while the root is over visible sand. A survival-first sweep found that
drag coefficients from 40 through 120 N per (m/s) all preserved survival over
eight 20 s headings while reducing speed progressively. Mild compliance also
remained upright when combined with those drag values. Deep compliance was
unstable even with drag and is rejected.

Use `--surface-linear-drag 120` plus the mild contact settings from
`scripts/run_unitree_sand_survival_sweep_local.sh` for the strongest tested
stable slowdown. In target-layout evaluation drag is spatially gated by the
final visible material. This setting intentionally prioritizes staying upright
over command tracking; lower drag remains available when mobility matters.

## Geometry density and capability

The generated presets now oversample bidirectional ramps and stairs relative
to rough ground and rubble. A separate controlled benchmark covers ramps from
5 to 30 degrees and symmetric stairs from 5 to 25 cm per step. The existing
flat-only locomotion checkpoint crossed none of the tested structures; runs
that remained upright on steeper ramps had stalled at the entrance and are not
counted as traversal successes.

Static obstacle half-extents now range from 0.32 to 1.15 m and moving-obstacle
half-extents from 0.25 to 0.72 m, with independently randomized heights. Moving
obstacles sample peak speeds from 0.65 to 1.8 m/s. Their sweep amplitude is
coupled inversely to speed, decreasing from 1.4 to 0.45 m, and both the speed
and resulting period are stored in each layout's metadata.

Render the default six inspection variants:

```bash
cd /home/benjamin/thesis && \
./scripts/render_unitree_target_terrains_local.sh
```

Audit all images, videos, and metadata:

```bash
cd /home/benjamin/thesis && \
/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/audit_unitree_target_terrains.py
```

Use `balanced` as the initial target candidate. `traversal` emphasizes terrain
transitions, while `navigation` emphasizes static and moving obstacle routing.
