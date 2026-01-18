# PCCC & SWCNT Visualization App

A Streamlit application for visualizing Partition Coefficient (PCCC) and UV-Vis absorption data for Single-Walled Carbon Nanotubes (SWCNTs).

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

## How to Run Locally

1.  **Clone the repository** (if using git):
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## Deployment

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):

1.  Push this code to a GitHub repository.
2.  Log in to Streamlit Cloud.
3.  Click "New app", select the repository, branch (usually `main`), and main file path (`app.py`).
4.  Click "Deploy".

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
