# PCCC & SWCNT Visualization App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/)

A Streamlit application for visualizing Partition Coefficient (PCCC) and UV-Vis absorption data for Single-Walled Carbon Nanotubes (SWCNTs).

## 🚀 Live Demo

**Access the running app here:**  
👉 **[https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/](https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/)**  
*(Deployment verified)*

## ⚠️ Scope & Disclaimer

Please note the following important limitations and contexts for this tool:

1.  **Simulated Spectra**: The UV-Vis plots are **simulated using Voigt profiles** for visualization and educational purposes only. They do **not** represent experimentally measured raw spectra.
2.  **Peak Variations**: Semiconducting SWCNT peak positions ($S_{11}, S_{22}$) are sourced from **Weisman et al. (2003)** measured in SDS. These values may exhibit systematic blueshifts or differences compared to data measured in DOC or in organic solvents with PFO.
3.  **Limited Dataset**: The PCCC dataset is a **small-sample demo** derived only from the publicly available parameters in the ACS Nano 2015 paper. If a specific chirality is missing, it indicates the data is **not currently indexed** in this demo, rather than a software error.

## Features

- **Chirality Selection**: Choose specific (n,m) SWCNTs to visualize.
- **PCCC vs SDS**: View Partition Coefficient curves modeled with Hill equations.
- **Simulated UV-Vis**: Generate absorption spectra using Voigt profiles for both Semiconducting and Metallic tubes.
- **Data Tables**: Detailed properties (diameter, chiral angle, E11/E22 or M11) for selected nanotubes.

## Data Sources

This tool uses only publicly available data from the following literature:

*   **PCCC Parameters**:
    > *Precise Partitioning of Metallic Single-Wall Carbon Nanotubes and Enantiomers through Aqueous Two-Phase Extraction*  
    > **Han Li**, et al. *ACS Nano*, 2015. [DOI: 10.1021/acsnano.5c00025](https://pubs.acs.org/doi/full/10.1021/acsnano.5c00025)

*   **Semiconducting SWCNT Data (SDS)**:
    > *Dependence of Optical Transition Energies on Structure for Single-Walled Carbon Nanotubes in Aqueous Suspension: An Empirical Kataura Plot*  
    > R. B. Weisman & S. M. Bachilo. *Nano Lett.*, 2003. [DOI: 10.1021/nl034428i](https://pubs.acs.org/doi/10.1021/nl034428i)

*   **Metallic SWCNT Data**:
    > *Fundamental optical processes in armchair carbon nanotubes*  
    > *Nanoscale*, 2013. [DOI: 10.1039/C2NR32769D](https://pubs.rsc.org/en/content/articlelanding/2013/nr/c2nr32769d)

## Data Schema

The CSV files in `data/` follow this structure (missing values denoted by empty cells or `-`):

| File | Key Columns | Unit / Description |
| :--- | :--- | :--- |
| **pccc_parameters.csv** | `pccc`, `n_h` | Partition coefficient parameters (Hill curve) |
| | `pccc_sigma`, `n_h_sigma` | Standard deviation of parameters |
| | `doc_pct`, `sc_pct` | Concentration in % |
| **swcnt_info_...** | `diameter` | Tube diameter (nm) |
| | `chiral_angle` | Chiral angle (degrees) |
| | `wl11` / `M11` | Peak Wavelength (nm) |
| | `rbm` | Radial Breathing Mode ($cm^{-1}$) |

## 🛠️ Conda Quickstart

To run this project locally, we recommend using Conda:

1.  **Create Environment**:
    ```bash
    conda create -n atpe_app python=3.11
    conda activate atpe_app
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Application**:
    ```bash
    streamlit run app.py
    ```

## How to Cite

If you use this tool or data in standardizing your research, please cite the original **Han Li et al. (ACS Nano 2015)** paper regarding the PCCC parameters.

For the software itself:
> **Han Li**. *ATPE PCCC Visualization Tool*. GitHub Repository, 2024. [https://github.com/HLi2022/atpe-pccc-demo](https://github.com/HLi2022/atpe-pccc-demo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
**Open Source Principle**: This repository contains only code and data derived from publicly published works. No private or restricted data is included.
