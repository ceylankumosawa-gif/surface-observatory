# NASA access on the existing Hetzner server

Prepared on 9 September 2026 for the option B research workflow. Account registration alone does not authenticate the server. Run the following command in your own terminal, not in a chat message:

```sh
ssh -t hetzner2 lst-earthdata-login
```

Enter your Earthdata username and password at the prompts. Password input is hidden. The helper sends the password only to NASA's documented HTTPS `find_or_create_token` endpoint, with redirects disabled. It stores the returned revocable bearer token; it does not save the username or password. No `.netrc` or shell environment credentials are created. NASA currently documents a 60-day token lifetime; the helper prints the returned expiry date. An existing token may expire sooner. [NASA token documentation](https://urs.earthdata.nasa.gov/documentation/for_users/user_token).

The helper then requests just the first 16 bytes of one ECOSTRESS TIFF and one ASTER TIFF to check protected-file access. These are file headers, not temperature pixels; this does not open or evaluate reserved test labels. It never follows redirects carrying the password, and sends the bearer token only to the exact NASA LP DAAC download host. It may follow a signed S3 asset redirect without forwarding the token. A successful login and successful access to both datasets are reported separately.

If NASA rejects programmatic password login, generate a token in your own [Earthdata profile](https://urs.earthdata.nasa.gov/), then paste it into this hidden terminal prompt:

```sh
ssh -t hetzner2 lst-earthdata-login --token
```

Never paste a token or password into chat, a command argument, a public website field, a report or source code. To repeat the access check without displaying credentials:

```sh
ssh hetzner2 lst-earthdata-login --check
```

If access is unverified, sign in to Earthdata Search and review the NASA application authorization prompt for LP DAAC. Then retry `--check`. A temporary network failure, a replaced example file, or a service outage can also fail a file probe; a saved login does not by itself prove data access. Share only the helper's status messages when troubleshooting.

## Storage and operation

- The root-owned launcher is `/usr/local/bin/lst-earthdata-login`; its reviewed source is `deploy/lst-earthdata-login` and `deploy/earthdata_login.py`.
- The helper runs as dedicated system user `lstdata`, with no login shell.
- Token storage is `/var/lib/lst-data/earthdata/credential.json`, mode `0600`, under directories mode `0700`, owned by `lstdata`. It is deliberately outside the repository, source cache, web outputs and static website.
- The public website user `lstweb` cannot traverse the credentials directory. Credentials are for controlled research acquisition, not public request parameters.
- Routine agent checks must use the helper's `--check` status; never print, copy, back up into the project, or inspect the credential file in tool output.
- Acquisition code may read the token in the research process and attach it only to the approved NASA data host. It must not expose it in exceptions, query strings, logs or output manifests.
- Expiry or revocation requires another private login. No background renewal, scheduled download, model training or cloud provisioning is started by this helper.

No Earthdata Python package installation was needed for this login step; it uses the project's existing `requests` dependency. Data access is HTTPS from Hetzner and does not require an AWS account. [NASA HTTP access example](https://nasa.github.io/ECOSTRESS-Data-Resources/python/how-tos/how_to_direct_access_http_ecostress_cog.html).

## Verification

The helper's seven tests passed on Hetzner before installation: password exclusion from persisted content, private directory/file modes, disabled password redirects, removal of bearer credentials on signed S3 redirects, rejection of unexpected hosts/protocols/ports, a 16-byte read limit, and secret-free login failures. The launcher ran successfully as `lstdata`, and a filesystem permission check verified that `lstweb` could not read the token directory. At installation, no credentials were present and neither dataset was yet authenticated. Later access status must be checked separately; setup is not a claim of successful authenticated acquisition.
