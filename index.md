# Project Genesis: The Oracle Awakens

## Experiment Summary
In this project, we developed a Deep Autoencoder (The Oracle) to detect anomalies in physical signal flows, specifically focusing on RC-filter data. 

### Key Milestones:
1. **Architecting the Oracle**: Built a custom bottleneck network using the Keras Subclassing API.
2. **Cloud Compute Awakening**: Deployed the model to Google Colab, utilizing TPUs and JAX/XLA for accelerated training.
3. **Anomaly Detection**: Trained on normal signal data for 30 epochs. Successfully identified injected anomalies using Mean Absolute Error (MAE) reconstruction loss.
4. **Agentic Refactoring**: Upgraded the architecture from Dense layers to **1D Convolutional Layers** (Conv1D and Conv1DTranspose) to better capture local temporal patterns in the time-series data.

## Anomaly Detection Results
Below is the reconstruction loss plot showing the "normal" physics baseline and the undeniable spike where the anomaly was injected.

![Reconstruction Loss](reconstruction_loss.png)

*The red dashed line represents our automated Anomaly Threshold.*

## Week 5: The Fabric of Reality (Physics-Informed Neural Networks)
I have transcended classical grid-based solvers by implementing a **Physics-Informed Neural Network (PINN)**. Instead of slicing space and time into rigid meshes, the AI now internalizes the laws of thermodynamics directly.

### Milestones:
1. **Shattering the Grid**: Replaced FDM meshes with mesh-free JAX sampling (5,000 collocation points).
2. **AutoDiff Engine**: Utilized nested `jax.grad` to embed the 1D Heat Equation ($\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$) into the loss landscape.
3. **Neural Surrogate**: Architected a Flax-based MLP with `tanh` activation for smooth, continuous physics gradients.
4. **Interactive Manifolds**: Rendered the predicted temperature field as an interactive 3D surface using Plotly.

---

### Project Links
*   **[Fabric Report (PINN Analysis)](docs/Fabric_Report.md)**
*   **[Interactive 3D Simulation](data/pinn_3d_fabric.html)**

*The Oracle is evolving. Physics and Code are one.*
