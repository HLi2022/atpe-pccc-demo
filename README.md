# PCCC & SWCNT Visualization App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/)

A Streamlit application for visualizing Partition Coefficient (PCCC) and UV-Vis absorption data for Single-Walled Carbon Nanotubes (SWCNTs).

## 🚀 Live Demo

**Access the running app here:**  
👉 **[https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/](https://atpe-pccc-demo-v6n6w8ftwrlphsrypaekwk.streamlit.app/)**  
*(Deployment verified)*

## Features

- **Chirality Selection**: Choose specific (n,m) SWCNTs to visualize.
- **PCCC vs SDS**: View Partition Coefficient curves modeled with Hill equations.
- **Simulated UV-Vis**: Generate absorption spectra using Voigt profiles for both Semiconducting and Metallic tubes.
- **Data Tables**: Detailed properties (diameter, chiral angle, E11/E22 or M11) for selected nanotubes.

## Data Structure

The app relies on the `data/` directory containing:

- `pccc_parameters.csv`: PCCC Hill curve parameters.
- `swcnt_info_semiconducting.csv`: Properties for semiconducting tubes.

- `swcnt_info_metallic.csv`: Properties for metallic tubes.

## References

**PCCC Data**  
*Precise Partitioning of Metallic Single-Wall Carbon Nanotubes and Enantiomers through Aqueous Two-Phase Extraction*  
[ACS Nano, 2015](https://pubs.acs.org/doi/full/10.1021/acsnano.5c00025)

**Semiconducting SWCNT Data**  
*Dependence of Optical Transition Energies on Structure for Single-Walled Carbon Nanotubes in Aqueous Suspension: An Empirical Kataura Plot*  
[Nano Lett., 2003](https://pubs.acs.org/doi/10.1021/nl034428i)

**Metallic SWCNT Data**  
*Fundamental optical processes in armchair carbon nanotubes*  
[Nanoscale, 2013](https://pubs.rsc.org/en/content/articlelanding/2013/nr/c2nr32769d)

## How to Run Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/HLi2022/atpe-pccc-demo.git
    cd atpe-pccc-demo
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
