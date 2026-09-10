# scripts/make_viewer.py
"""Run the zero-shot fly for 300 actions and render a scrubbable viewer."""
from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np

from flycraftax.brain import BrainParams
from flycraftax.data import load_connectome
from flycraftax.drive import build_drive
from flycraftax.env import base_state, make_env
from flycraftax.loop import build_agent, init_carry, make_step
from flycraftax.readout import load_readout
from flycraftax.retina import build_retina
from flycraftax.viewer import compose, make_fly_renderer, soma_xy

HTML = """<!doctype html><title>fly-craftax viewer</title>
<style>body{font-family:sans-serif;margin:1em}img{max-width:100%%}</style>
<img id=f src="frames/0000.png"><br>
<input id=s type=range min=0 max=%d value=0 style="width:80%%"> <button id=p>play</button> <span id=t>0</span>
<script>
const n=%d,f=document.getElementById('f'),s=document.getElementById('s'),t=document.getElementById('t');let timer=null;
function show(i){f.src='frames/'+String(i).padStart(4,'0')+'.png';s.value=i;t.textContent=i;}
s.oninput=()=>show(+s.value);
document.getElementById('p').onclick=()=>{if(timer){clearInterval(timer);timer=null;return;}timer=setInterval(()=>show((+s.value+1)%%n),100);};
</script>"""


def main(n_actions=300, warm=20, out=Path("outputs/viewer")):
    conn = load_connectome()
    readout, cfg = load_readout(conn)
    drive = build_drive(conn, build_retina(conn), cfg["max_hz"], lamina_mv=cfg["lamina_mv"])
    agent = build_agent(conn, drive, readout, BrainParams(w_syn=cfg["w_syn"]))
    env = make_env(1)
    render = jax.jit(make_fly_renderer())
    xy = soma_xy(conn)
    key, k0 = jax.random.split(jax.random.PRNGKey(0))
    carry = init_carry(agent, env, k0, 1) + (None,)
    step = make_step(agent, env, obs0=carry[0], keep_counts=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames, pending, vmax = [], [], None
    for i, k in enumerate(jax.random.split(key, n_actions)):
        st = base_state(carry[1])
        frame64 = np.asarray(render(jax.tree.map(lambda x: x[0], st)))
        carry, log = step(carry, k)
        pending.append((
            frame64, np.asarray(log["counts"][0]), np.asarray(log["z"][0]), int(log["action"][0]),
            dict(health=int(st.player_health[0]), food=int(st.player_food[0]),
                 drink=int(st.player_drink[0]), energy=int(st.player_energy[0])),
        ))
        if vmax is None and len(pending) < min(warm, n_actions):
            continue     # hold the first windows back until the colour scale is known
        if vmax is None:  # one scale for every frame, so colours compare across the run
            nz = np.concatenate([c[c > 0] for _, c, *_ in pending])
            vmax = (float(np.percentile(nz, 95)) if nz.size else 0.0) or 3.0
        for frame, counts, z, action, meters in pending:
            img = compose(frame, counts, xy, readout.groups, z, action, meters, readout.z_floor, vmax=vmax)
            imageio.imwrite(out / "frames" / f"{len(frames):04d}.png", img)
            frames.append(img[::2, ::2])
        pending.clear()
        if i % 50 == 0:
            print(i, flush=True)
    (out / "index.html").write_text(HTML % (n_actions - 1, n_actions))
    imageio.mimsave(out.parent / "viewer.gif", frames, duration=100)  # imageio 2.37: ms, so 10 fps
    print(out / "index.html", out.parent / "viewer.gif")


if __name__ == "__main__":
    main()
