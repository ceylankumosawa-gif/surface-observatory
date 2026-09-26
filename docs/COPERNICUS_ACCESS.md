# Copernicus access for Hetzner research

The ERA5-Land ARCO service requires a Copernicus Climate Data Store personal access token. NASA Earthdata credentials do not provide this access.

1. Sign into [CDS](https://cds.climate.copernicus.eu/profile) and copy the Personal Access Token from your profile.
2. In your own terminal, run `ssh -t hetzner2 lst-cds-login`.
3. Paste only the token into the hidden prompt. It is normal for no characters to appear. Press Enter.
4. A successful connection prints “Copernicus connected.” Share that status only, never the token.

If dataset access is denied, visit the [ERA5-Land download form](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download) and accept its licence, then retry. If the key is rejected, copy the current token from the profile and retry. The helper does not request or store your account password.

The verified key is saved atomically at `/var/lib/lst-data/copernicus/credential.json`, owned by the `lstdata` research account, with directory mode 0700 and file mode 0600. It is outside the repository and website. The administrator can access it; the website service cannot. The helper sends it only to the fixed HTTPS ECMWF ARCO metadata endpoint and refuses redirects. It verifies metadata without downloading thermal labels, limits the response to 1 MiB and prints only safe status messages. Failed verification does not replace an existing key.

Check a saved connection with `ssh hetzner2 lst-cds-login --check`. Run the login command again to replace a revoked or expired token. Manage token revocation through the Copernicus account. Do not put credentials in chat, source files, download URLs, command arguments, notebooks or reports.

The extraction adapter accepts `--credential-file /var/lib/lst-data/copernicus/credential.json` and independently enforces private file permissions, fixed ECMWF destinations and no redirected credentials. It records data provenance and request budgets without recording the token.

Sources: [CDS API setup](https://cds.climate.copernicus.eu/how-to-api), [ERA5-Land ARCO access](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data). Helper installed and security checks passed on Hetzner on 2026-09-11.
