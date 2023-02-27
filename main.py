#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

from ColloidSim import ColloidSim, ColloidSimParams


if __name__ == "__main__":

    # Set up basic 2-particle test case: particles going towards each other at the same speed
    params = ColloidSimParams(
        n_particles = 2,
        posns_0 = np.array([[0.05, 0.01, 0.005], [-0.05, 0, 0]]).T,
        vels_0 = np.array([[-1, 0, 0], [1, 0, 0]]).T
    )
    sim = ColloidSim(params)

    sim.simulate()

    st.write("Simulation Results")
    t = st.slider("Time", 0, params.n_steps - 1, 0)
    fig = go.Figure(go.Scatter3d(
        x=sim.posns[t, 0, :],
        y=sim.posns[t, 1, :],
        z=sim.posns[t, 2, :],
        mode="markers",
        marker={"color": px.colors.qualitative.Dark24, "size": 5}
    ))

    fig.update_layout(
        scene_aspectmode="cube",
        height=800,
        width=800,
        scene={
            "xaxis": {"range": [-params.box_x/2, params.box_x/2]},
            "yaxis": {"range": [-params.box_y/2, params.box_y/2]},
            "zaxis": {"range": [-params.box_z/2, params.box_z/2]},
        },
        uirevision="hi"
    )

    st.plotly_chart(fig)
