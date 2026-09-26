# Third-party notices and data attribution

The Apache License 2.0 in [LICENSE](LICENSE) applies to this project's original software and documentation. It does not replace the licences of third-party packages, boundary data, remotely accessed datasets, or map services. The initial source release does not include trained model weights, raw satellite or weather archives, private credentials, or installed dependency trees.

The public research CSV/JSON files contain project-generated summaries and provenance. They are not a redistribution of the underlying observation archives. Retain their source citations and qualifications when reusing them. Dataset attribution does not imply endorsement by a data provider.

## Included boundary data

`pilot/london_boroughs_27700.json` and `pilot/london_boroughs.geojson` originate from the Greater London Authority's London Borough layer. These files, and boundary geometry derived from them, remain under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), rather than Apache-2.0.

The [GLA source service](https://gis.london.gov.uk/arcgis/rest/services/apps/webmap_context_layer/MapServer/3?f=pjson) supplies this attribution:

> Credits: Greater London Authority - Contains public sector information licensed under the Open Government Licence v3.0

The project uses simplified boundary geometry (50 m tolerance) and transforms coordinates between British National Grid and WGS84 for its planning and display files. These are project modifications, not an authoritative replacement for the source boundaries. Retain the attribution and licence link with copies and derived boundary files.

## Software dependencies and artwork

Dependencies are installed separately using [pyproject.toml](pyproject.toml) and [web/package.json](web/package.json); the JavaScript resolution is recorded in [web/package-lock.json](web/package-lock.json). Their own licences and copyright notices continue to apply. When distributing a compiled website, container, wheel, or other bundle that includes dependency code or data, include the licence and notice files supplied by the exact resolved packages, including transitive dependencies. This summary is not a substitute for those files.

| Component | Upstream terms and relevant distinction |
| --- | --- |
| React and React DOM | [MIT](https://github.com/react/react/blob/main/LICENSE); copyright Meta Platforms, Inc. and affiliates. |
| MapLibre GL JS | [BSD-3-Clause and included component notices](https://github.com/maplibre/maplibre-gl-js/blob/main/LICENSE.txt). Its upstream notice also covers inherited Mapbox GL JS, glfx.js and d3-color code. |
| Lucide React icons | [ISC; Feather-derived icons additionally carry MIT notices](https://lucide.dev/license). Preserve the notices from the installed Lucide package. The application imports icons from this package. |
| pvlib Python | [BSD-3-Clause](https://github.com/pvlib/pvlib-python/blob/main/LICENSE). Solar-position calculations call pvlib; this project does not vendor the separately distributed NREL `spa.c` implementation. |
| timezonefinder 8.2.4 | [MIT software; ODbL boundary data](https://github.com/jannikmi/timezonefinder/tree/8.2.4#license), derived from timezone-boundary-builder. A runtime distribution containing its boundary database must preserve its separate data licence and attribution; the Apache software licence does not cover that database. |
| Matplotlib | [Matplotlib's own licence and included third-party notices](https://matplotlib.org/stable/project/license.html). Used as a plotting dependency. |
| Remaining Python and web dependencies | Retain each installed package's own licence, including the numerical, geospatial, HTTP, application-server and build-tool dependencies declared in the manifests. Installing these packages does not relicense them under this project's Apache grant. |

`web/public/model-flow.png` is a project-generated diagram produced by `src/lst_pilot/report.py`, not an external photograph or downloaded illustration. The supplied copy matches the project's generated research diagram. The favicon is project SVG geometry. No external font binaries are included in `web/public`.

The browser's solar calculation cites [NOAA's solar equations](https://gml.noaa.gov/grad/solcalc/calcdetails.html) in `web/src/local-time.ts`. Time-zone identifiers there cite the [IANA time-zone database](https://data.iana.org/time-zones/tzdb/zone.tab).

## Scientific data sources

The source-specific credits in [web/catalog.json](web/catalog.json) and output provenance should accompany applicable derived outputs. Access through a delivery service does not transfer ownership of the underlying dataset.

| Source | Attribution and applicable terms |
| --- | --- |
| Landsat Collection 2 | Landsat data courtesy of the U.S. Geological Survey. [USGS reuse guidance](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data) identifies Landsat data as public domain. Microsoft Planetary Computer is a delivery service. Preserve collection/version and scene identifiers. |
| NASA Earthdata products, including ECOSTRESS, ASTER, MODIS and VIIRS | Cite the exact collection/version, dataset DOI, granule identifiers and processing changes recorded in each source receipt. Follow the applicable collection's data-use terms and acknowledgement guidance; the repository's software licence grants no additional rights over NASA-hosted or partner-supplied data. Authentication or AppEEARS access is not a new data licence. |
| NOAA station and radiation observations | Credit NOAA NCEI for GHCNh/ISD and NOAA's SURFRAD network for radiation observations, retaining station IDs and source QC. The [GHCNh dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01688) specifies CC0-1.0 and its dataset citation. Do not assume that every contributing agency or separately sourced station archive has identical terms. |
| BSRN/PANGAEA and OzFlux station records | Retain the authors, site investigators, exact dataset DOI/version and licence attached to each downloaded record. Their project-specific radiometric summaries do not relicense the underlying station archives. No raw station archive is included in this source release. |
| ERA5 and ERA5-Land | Credit ECMWF / Copernicus Climate Change Service and retain the dataset DOI/version. The [ERA5 catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and [ERA5-Land catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) govern the meteorological data. Applicable modified outputs should retain: “Contains modified Copernicus Climate Change Service information.” The data are separate from Apache-licensed software used to retrieve them. |
| Google ARCO-ERA5 | Credit Carver and Merose (2023), ARCO-ERA5, and the underlying ECMWF/C3S data. The [ARCO-ERA5 repository](https://github.com/google-research/arco-era5) distinguishes its software from the underlying ERA5 data. |
| Open-Meteo | Weather data by Open-Meteo.com; underlying model data by their identified providers. [Data are CC BY 4.0; API-service terms are separate](https://open-meteo.com/en/terms). In particular, the free API is restricted to noncommercial use and has request limits. An Apache-licensed client does not grant unrestricted or commercial access to that service. |
| Copernicus DEM GLO-30 | Preserve the [Copernicus WorldDEM-30 licence](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf). Project derivative credit: “produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved”. See the [official dataset description](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM). |
| ESA WorldCover 2020 v100 / 2021 v200 | [CC BY 4.0 and product acknowledgement guidance](https://esa-worldcover.org/en/data-access). Preserve the corresponding year: “© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium” or the equivalent 2021 credit. Record aggregation and masking changes. |
| Köppen–Geiger 1991–2020 raster | Credit Beck et al. (2023), High-resolution Köppen–Geiger maps, [dataset DOI](https://doi.org/10.6084/m9.figshare.21937571.v1), and the NatCap COG conversion. The project's cited [dataset record](https://api.figshare.com/v2/articles/21937571) identifies CC0-1.0; the paper's licence must not be substituted for the dataset licence. |
| Natural Earth | [Public-domain data](https://www.naturalearthdata.com/about/terms-of-use/). Credit Natural Earth when its cartographic boundaries are used. |

## Map services

The interactive basemap uses OpenStreetMap. Keep a visible linked **© OpenStreetMap contributors** attribution and comply with the [ODbL/copyright information](https://www.openstreetmap.org/copyright) and the separate [public tile-service policy](https://operations.osmfoundation.org/policies/tiles/). Public standard tiles must not be bulk-prefetched or packaged for offline use. No map-tile archive is included here.

Historical examples that refer to NASA GIBS imagery must retain the imagery's NASA/provider credits. A MapLibre software licence does not grant rights to a third-party hosted basemap, imagery, logo, or service.

## Scope of these notices

These attributions describe included boundary data and the upstream dependencies and data sources used by the project. They do not modify [LICENSE](LICENSE) or imply that excluded raw archives or model weights are being offered for download. Preserve any applicable upstream `LICENSE` and `NOTICE` files when redistributing upstream material, as required by its licence and [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0).
