# Private AppEEARS access

The helper is ready for a private terminal login. Preparation and tests do not
authenticate with NASA. When AppEEARS access is needed, run:

```sh
ssh -t hetzner2 lst-appeears-login
```

Enter your existing Earthdata username and password in that terminal. The password
prompt is hidden; do not paste credentials into chat. The helper contacts only
`https://appeears.earthdatacloud.nasa.gov/api/login`, with verified HTTPS and no
redirects, proxy environment, automatic retries, subset requests or downloads.
It stores only the separate AppEEARS token, its type and expiry, and a saved time.
NASA documents a lifetime of approximately 48 hours. Re-run the same command to
renew it when needed. [NASA API authentication](https://appeears.earthdatacloud.nasa.gov/api/#authentication)

To check only the saved AppEEARS token's permissions, structure and expiry locally:

```sh
ssh hetzner2 lst-appeears-login --check
```

This check does not contact NASA or prove the token has not been revoked. It never
prints the token. A successful login atomically replaces an older AppEEARS token;
a failed login or write before replacement preserves it.

Earthdata profile bearer tokens are incompatible with AppEEARS. The existing
Earthdata credential is neither read nor changed by this helper.
[NASA download guide](https://github.com/nasa/AppEEARS-Data-Resources/blob/main/guides/How-to-bulk-download-AppEEARS-outputs.md)

The new token is stored at `/var/lib/lst-data/appeears/credential.json`, owned by
`lstdata`, mode `0600`, inside a `0700` directory. The launcher and installed
helper are root-owned at `/usr/local/bin/lst-appeears-login` and
`/usr/local/libexec/lst-appeears-login.py`. Installation does not create a token.
Writes use a private temporary file, file/directory synchronization and atomic
replacement; unsafe symlink, hardlink and ownership configurations are rejected.
Authentication errors never print server bodies, headers or exception details.

The installed helper uses Python isolated mode and runs only as the existing
research data user. The website user `lstweb` cannot traverse the private data
directory. Tests use fake sessions and temporary files on Hetzner, without reading
real credentials or making authentication requests.

Verified on Hetzner on 19 September 2026: 37 mock tests passed in 0.11 seconds
under a network-isolated service (2 CPU quota, 1 GiB limit, 120-second deadline;
31 MiB measured peak). The installed helper matches the repository source SHA256
`0315db5c3005eb32a1802cbc641c7300d8b76f153e5331980fee4c5700c94b99`.
Launcher/helper ownership and directory permissions were checked, and `lstweb`
was denied directory traversal. Only `--help` was invoked on the installed helper;
no real login, saved-token read, subset task or imagery request occurred. The token
file did not exist at installation closeout.
