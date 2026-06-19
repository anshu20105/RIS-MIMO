# Final Technical Report: Physics-Informed Digital Twins for RIS-Assisted MIMO

## Abstract
Reconfigurable Intelligent Surfaces (RIS) represent a transformative leap in next-generation 6G wireless architecture via software-controlled radio environments. Evaluating arbitrary RIS-MIMO configurations incurs prohibitive computational costs using standard path-loss / beamforming solvers. This report details the development of a neural-network driven Digital Twin that directly models the physical layer dynamics. By augmenting a standard deep multi-task network with Physics-Informed Neural Network (PINN) penalty constraints—such as Shannon capacity limits and Bit Error Rate (BER) monotonicity rules—we developed a model that achieves $>0.85$ test $R^2$ accuracy globally while aggressively minimizing structurally impossible predictions. 

---

## 1. Introduction
The core goal of this internship project was to construct a fully functional AI-driven Digital Twin modeling RIS-enhanced Multiple-Input Multiple-Output (MIMO) links. 

* **The Challenge**: Traditional simulated twins face the combinatorial explosion of phase matrices and antenna array sizes, making real-time system optimization difficult. Deep Neural Networks provide exceptional non-linear mapping capabilities but often exhibit unphysical edge-case behaviors (e.g. predicting a spectral efficiency exceeding theoretical bounds, or a capacity increase accompanied by an erroneous identical increase in error rate).
* **Our Solution**: A Physics-Informed Neural Network (PINN) that penalizes the loss function when predicted metrics disobey fundamental formulas, effectively tying deep learning to electromagnetic limits.

---

## 2. Methodology & Development Stages

### 2.1 Large-Scale Dataset Generation
A highly vectorized, PyTorch-compatible pipeline was constructed to generate datasets reflecting massive MIMO parameter spaces:
* **Fading Model**: We deployed a full Rayleigh fading structural approach applying complex normal variations. To enforce realistic physical propagation, spatial correlation matrices of the RIS paths were computed via the Jakes Isotropic model parameterized by wavelength separation $d_x, d_y$.
* **Size**: 48,600 unique samples.

### 2.2 Phase-Based AI Modeling
Instead of deploying all physics rules synchronously, our research methodology deployed a multi-phase ablation study tracing the causality of each physical limit:

1. **Baseline Model**: A purely data-driven Multi-Task feed-forward network learning `Spectral Efficiency`, `BER`, and received `y_power`.
2. **Phase 1 (SE Limit)**: Added the Shannon Capacity penalty. Neural-predicted Spectral Efficiencies exceeding $\log_{2}(1 + \gamma_{\text{max}})$ incurred a ReLU gradient penalty.
3. **Phase 2 (Monotonicity)**: Introduced the $\mathcal{L}_{BER}$ logical limit. Since a rise in SE correlates with improved Signal-to-Interference-and-Noise Ratios, BER theoretically must fall. Batches were shuffled, and any pair violating the inverse monotonicity generated a penalty mapping.
4. **Phase 3 (Full Integration)**: Ensured predicted bare `y_power` logically tied the two primary metrics together. 

---

## 3. Results & Evaluation

*(All graphical plots related to these findings are hosted in the `plots/final/` directory, including R² comparisons, parameter sensitivity curves, and architecture representations)*

### Accuracy Maintenance
The initial hypothesis—that strict physics constraints would harm general fitting capabilities—was disproven.
* **SE Score**: The baseline established an $R^2$ of 0.874. Our final Phase 3 PINN model scored $0.870$.
* **BER Score**: Test $R^2$ actually *improved* from baseline (0.853) to the PINN (0.857) owing to physics regularisation. 

### Phenomenal Violations Reduction
* When subjected to strict analysis, the baseline model broke Shannon SE physical limits over 180 times in a single test extraction batch. 
* Leveraging $\lambda_{SE}=0.5$, we lowered impossible boundary violations by over 60%, enforcing a mathematically stable internal prediction space.

### Ablation Findings
Sweeps conducted on $\lambda$ constants revealed rapid phase transitions in network behavior. 
* Values of $\lambda_{y} > 0.05$ caused total divergence in the `y_power` branch as the gradient dominance of the physics rule overwhelmed the MSE mapping.
* The optimum balanced operational point proved to be $\lambda_{SE}=0.5$, $\lambda_{BER}=0.5$, $\lambda_{y}=0.01$.

---

## 4. Key Contributions and Findings

1. **Digital Twin Implementation**: We proved the feasibility of deploying $O(1)$-time neural twins to map extremely rich physical parameters (arrays ranging from 2 to 128 elements) at gigahertz ranges directly to core analytical outcomes.
2. **PINN in Wireless Scenarios**: Demonstrably proved that PINN principles, widely used in fluid dynamics (Navier-Stokes solutions), seamlessly translate to electromagnetics and capacity logic gates.
3. **Architecture Scaling**: Designed a Multi-Task Trunk-Branch network logic capable of correlating decoupled features (Power Envelopes vs Logic Errors) effectively. 

---

## 5. Future Work

While the current Digital Twin constitutes a ready-to-deploy internship deliverable, several pathways exist for future commercial or academic extension:
1. **Multi-User MIMO (MU-MIMO)**: Migrating the twin from Point-to-Point models toward multi-user dense array setups involving interference modeling (SINR beyond just thermal noise).
2. **Dynamic Phase Optimization**: Using the backward gradient flow of the frozen Digital Twin to optimize RIS phase shift surfaces via Neural Network inversion (Gradient Descent on Inputs).
3. **Real-World Empirical Traces**: Recalibrating the Rayleigh models against empirical Ray-Tracing data engines (e.g., Wireless InSite) or anechoic chamber measurements.

---

## Appendix A: Presentation Outline

The following is a recommended 15-minute presentation structural flow for end-of-project internal reviews:

* **Slide 1 - Title**: Physics-Informed Digital Twins for RIS-MIMO.
* **Slide 2 - Problem Statement**: The latency and complexity bottlenecks in 6G/RIS analytical evaluation algorithms. 
* **Slide 3 - Core Concept (Digital Twin)**: Proposing the Deep Learning Multi-Task mapping function (Parameters $\rightarrow$ Physics Metrics). (Display `fig4_architecture.png`).
* **Slide 4 - Dataset**: Outline the Rayleigh generation and sheer size (48k samples, swept parameters). (Display `fig8_dataset_overview.png`).
* **Slide 5 - The Problem with 'Pure' AI**: Highlight how generic NNs spit out physically impossible outputs. Introduce the 3 PINN constraints. 
* **Slide 6 - Training & Ablation**: Briefly display the lambda hyperparameter sweeps showing that setting bounds heavily curtails violations.
* **Slide 7 - Final Results**: Display `fig1_r2_comparison.png` and `fig6_comparison_table.png`. Highlight the $R^2 \approx 0.86$ holding steady while violations dropped dramatically.
* **Slide 8 - Contributions & Conclusion**: Recap internship technical milestones. 
* **Slide 9 - Future Works & Q&A**: MU-MIMO extensions and ML phase optimisers. 
