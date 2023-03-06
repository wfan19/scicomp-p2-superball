import numpy as np

from ColloidSim import ColloidSim

import plotly.graph_objects as go

def plot_results(sim: ColloidSim, fig=None, show_fig = False):
    if fig is None:
        fig = go.Figure()
        show_fig = True

    posns_0 = sim.posns[1, :, :] # Use posns from first timestep because overlaps have been resolved
    sim.posns[0, :, :] = sim.posns[1, :, :]
    displacements = sim.posns - posns_0
    distances = np.linalg.norm(displacements, axis=1)
    mean_squared_dists = np.mean(np.square(distances), axis=1)
    
    fig_mean_squared_dists = fig.add_trace(go.Scatter(
        x=sim.times,
        y=mean_squared_dists,
        name=f"N: {sim.params.n_particles}"
    ))
    fig_mean_squared_dists.update_xaxes(title="Time (s)", type="log")
    fig_mean_squared_dists.update_yaxes(title="Mean squared displacement", type="log")

    if show_fig:
        fig_mean_squared_dists.show()
