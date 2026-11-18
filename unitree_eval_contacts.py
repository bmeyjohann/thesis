"""Read-only contact reduction for single-environment offline evaluation."""
import warp as wp


@wp.kernel
def accumulate_contacts(n: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                        dist: wp.array(dtype=wp.float32), world: wp.array(dtype=wp.int32),
                        kind: wp.array(dtype=wp.int64), metrics: wp.array(dtype=wp.float32)):
    hit = int(0)
    depth = float(0.)
    for i in range(n[0]):
        a, b = geom[i][0], geom[i][1]
        if world[i] == 0 and a >= 0 and b >= 0 and dist[i] <= 0.:
            if (kind[a] == 1 and kind[b] == 2) or (kind[a] == 2 and kind[b] == 1):
                hit = 1
                depth = wp.max(depth, -dist[i])
    metrics[0] += float(hit)
    metrics[1] = wp.max(metrics[1], depth)
