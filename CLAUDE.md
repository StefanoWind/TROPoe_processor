# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A wrapper around David Turner's **TROPoe** thermodynamic retrieval algorithm (distributed as a docker/podman/apptainer container). TROPoe retrieves temperature and water vapor profiles from ground-based infrared spectrometer (IRS) radiances — ASSIST/AERI instruments — combined with cloud base height (CBH) and surface met data. This repo does not implement the retrieval itself; it downloads/reformats heterogeneous raw instrument data from different sites/networks into the file layout and VIP control file TROPoe expects, launches the TROPoe container per day, and produces QC/diagnostic plots of the output.

## Commands

There is no build system, linter, or test suite — this is a set of orchestration/preprocessing scripts run directly with Python.

- Set up the environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. `lidargo` (NREL/FIEXTA, `lidargo` subdirectory) has no PyPI release and is installed straight from GitHub via the `requirements.txt` entry; `doe-dap-dl` is on PyPI. Note the TROPoe container image itself (docker/podman/apptainer) is a separate dependency, not installed via pip.

- Run the full pipeline over a date range for a site:
  `python tropoe_launcher.py <site> <sdate:YYYYMMDD> <edate:YYYYMMDD> <serial|parallel> <config.yaml>`
  e.g. `python tropoe_launcher.py s40_rt 20260223 20260716 serial config_corsair.yaml`
  Running with no arguments uses the hardcoded defaults at the top of `tropoe_launcher.py` (convenient for interactive/IDE debugging — edit those in place rather than adding new CLI plumbing).
- Preprocess a single day only (no TROPoe run): `python tropoe_inputs.py <site> <date> <config_path> <tmpdir>`
- Launch the TROPoe container directly once inputs exist: `./run_tropoe_ops.sh <yyyymmdd> <vip_file> <prior_file> <shour> <ehour> <verbosity> <data_path> <tmp_path> <image_name> <image_type>` (`image_type` is `docker`, `podman`, or `apptainer`).
- Check how complete/populated a batch of output files is: `python check_progress.py <folder>`
- Run the IRS PCA noise filter standalone: `python utils/run_irs_nf.py <start> <end> <idir> <sdir> <odir> <irs_type:{dmv2cdf,assist,arm}> --create --apply`
- Compare CBH retrieved from lidar vs. ceilometer for a site pair: `python compare_cbh.py <channel_ceil> <channel_lid> <sdate> <edate> <config.yaml>`

## Architecture

### Pipeline flow (`tropoe_launcher.py`)
For each day in the range: `tropoe_inputs.py` runs as a subprocess to build TROPoe's inputs, `run_tropoe_ops.sh` launches the TROPoe container, and on success the output is post-processed (QC-plotted via `tropoe_utils.plot_temp_wvmr`, added to `data/processed-<site>.txt` so reruns skip completed days, temp dir cleaned up). `option` selects `serial` or `multiprocessing.Pool` (`parallel`) execution across days. Before processing, raw IRS/CBH/met data is bulk-downloaded from the DOE A2e data catalog (`a2e.energy.gov`, via `doe_dap_dl.DAP`) unless the configured channel is already a local `raw` channel.

### Per-day preprocessing (`tropoe_inputs.py`, backed by `utils/tropoe_utils.py`)
1. If the configured IRS channel contains `"raw"`, convert vendor-raw ASSIST files (and, if the CBH channel is a raw Halo lidar, `.hpl` scans via the external `lidargo` package) into the standard `00`-level layout first.
2. Copy/rename `00`-level ASSIST channel1 (`ch1`) and summary (`sum`) files into per-run temp directories, with an optional prefilter (QC on responsivity/imaginary radiance) and optional hatch-flag override (brightness-temperature-based proxy when the instrument's own hatch flag is unreliable).
3. Run the IRS PCA noise filter (`utils/run_irs_nf.py` / `utils/irs_nf/`) over a trailing `N_days_nfc[site]`-day window to denoise radiances before retrieval (skipped when the window is 1 day).
4. Derive CBH: from Halo lidar (`utils/cbh_halo.py`, attenuated-backscatter gradient method, Newsom et al. 2019) or from a Vaisala ceilometer (`tropoe_utils.extract_cbh_ceil`), written as a `*cbh*` product.
5. Derive surface met (temperature/pressure/RH) with site-specific parsers for the handful of supported formats (`rhod` CSV, `barg` raw CSV or `a0` netCDF), written as a `*sel*` product.
6. Fill in a VIP control-file template (`configs/vip_<site>*.txt`, with a `{date}` placeholder) for this day/site and write the diagnostic input-data plot.
7. Missing CBH/met data is either fatal or tolerated per-site via `allow_no_cbh`/`allow_no_met` in the config, and a max-allowed data gap check (`max_data_gap`) guards against silently retrieving over large outages.

### Data channel naming convention
Everything under `data/` follows `<network>/<site>.<instrument>.<zXX>.<level>`, e.g. `corsair/s40.assist.z01.00`. Levels seen in this repo: `raw` (vendor-native files), `00` (ingested/renamed), `cbh`/`sel` (derived CBH/met products consumed by TROPoe), `c0` (final TROPoe retrieval output, named by swapping `assist`→`assist.tropoe` and `00`→`data_level_output`). Code frequently derives one channel path from another via string slicing/replacement on this convention (e.g. `channel[:-2]+'cbh'`) rather than an explicit mapping — when adding a new site/instrument, keep filenames consistent with this convention or these derivations will silently point at the wrong directory.

### Config files
- `configs/config_*.yaml`: per-site settings keyed by site name — channel paths for IRS/CBH/met, `N_days_nfc` (noise-filter window length, also controls whether the PCA filter runs at all), prior file/climatology selection, QC thresholds (`max_gamma`, `max_rmsa`, `max_time_diff`, `max_data_gap`), `allow_no_cbh`/`allow_no_met`, and which container runtime/image to use.
- `configs/vip_*.txt`: TROPoe VIP control-file templates (one per site), with `{date}` substituted at runtime.
- `configs/config_corsair_format.xlsx`: `lidargo` formatting config for raw Halo lidar ingestion.
- `prior/Xa_Sa_datafile.<location>.55_levels.month_NN.cdf`: monthly climatological prior covariance files used when a site has no dedicated `prior_file`.

### Logging and idempotency
Every day/site run gets its own log file at `log/<site>/<date>.log` (`utils/utils.py` `create_logger`/`close_logger`). Completed days are recorded in `data/processed-<site>.txt`; `process_day` skips any date already listed there, so reprocessing a day requires removing its entry from that file.

### Standalone/ad hoc scripts
`check_progress.py`, `compare_cbh.py`, and `copy_rename_files.py` are auxiliary tools (progress inspection, CBH cross-validation between instruments, one-off local file renaming) rather than part of the main pipeline; `copy_rename_files.py` in particular has hardcoded source/destination paths meant to be edited per use.
