import numpy as np

from ColloidSim import ColloidSim

import plotly.graph_objects as go

def plot_results(sim: ColloidSim):
    posns_0 = sim.params.posns_0
    displacements = sim.posns - posns_0
    distances = np.linalg.norm(displacements, axis=1)
    mean_squared_dists = np.mean(np.square(distances), axis=1)
    
    fig_mean_squared_dists = go.Figure(go.Scatter(
        x=mean_squared_dists,
        y=sim.times,
    ))
    fig_mean_squared_dists.update_xaxes(title="Time (s)", type="log")
    fig_mean_squared_dists.update_yaxes(title="Mean squared displacement", type="log")
    fig_mean_squared_dists.show()
