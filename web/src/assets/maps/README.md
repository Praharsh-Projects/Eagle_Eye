# Offline map data

`ne_110m_land.geojson` is the Natural Earth 1:110m land layer used to give
Eagle Eye maps geographic context without a network basemap.

- Source: https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_land.geojson
- Retrieved: 2026-07-23
- SHA-256: `9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9`
- License: Public domain. Natural Earth permits use in any type of project.
  See https://www.naturalearthdata.com/about/terms-of-use/

The layer is contextual cartography only. Eagle Eye does not derive, move, or
replace analysis coordinates with Natural Earth data.

`ne_110m_baltic_countries.geojson` is a compact extract of the Natural Earth
1:110m admin-0 country layer. It contains only Sweden, Finland, Estonia,
Latvia, Lithuania, Poland, Denmark, and Germany and retains only each
country's name, ISO alpha-2 code, and geometry.

- Source: https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson
- Retrieved: 2026-07-28
- SHA-256: `4444457082162c0af0d0fe5880760f29f94abaf564bc189ecdf95dd766525c28`
- License: Public domain under the same Natural Earth terms above.

The country layer is presentation context for the Overview. Runtime capability
data decides which countries are in the AIS scope and the number of historical
UN/LOCODE entries represented; the asset does not supply operational facts.
