# Evidence of human influence on Northern Hemisphere snow loss
Replication code for "Evidence of human influence on Northern Hemisphere snow loss" by Gottlieb and Mankin.

The repository contains three folders:
- `data/`, where processed data is stored.
- `figures/`, where .pdf (.jpg) outputs for the main (Extended Data) figures from the scripts are stored. For two Extended Data figures (5 and 10), neat spacing of the subplots was challenging in `python`, so panels were saved as separate .jpg files and arranged in Inkscape.
- `scripts/`, which contains the `python` code used to perform the analysis and generate the figures. Note that the files `swe_recons_ml.py`, `swe_fut.py`, and `runoff_model_ml.py`, which run the full factorial reconstructions/projections of SWE and runoff, are written to be run in parallel on Dartmouth's High Performance Computing Discovery cluster.

For any questions about the code or requests for data, please contact alexander (dot) r (dot) gottlieb (dot) gr (at) dartmouth (dot) edu.
