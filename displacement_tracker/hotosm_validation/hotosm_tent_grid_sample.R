library(terra)
library(sf)
library(sampling)

#### USER OPTIONS ####

# sample size
n <- 500

# proportion of sample with zero-tent cells
zero_fraction <- 0.05

# date of tent data
date_tents <- '2026-08-05'

# planet images
planet_images <- c(
  file.path(
    '2026 08 (Aug)',
    'al_mawasi_deir_el_balah_nuseirat_20260805_125029_ssc10_u0002_visual_clip.tif'
  ),
  file.path(
    '2026 08 (Aug)',
    'gaza_city_jabalia_beit_lahiya_20260805_125029_ssc10_u0001_visual_clip.tif'
  ),
  file.path(
    '2026 08 (Aug)',
    'rafah_al_mawasi_khan_yunis_20260805_125029_ssc10_u0003_visual_clip.tif'
  )
)


#################################

#---- directories ----#

# working directory
data_dir <- file.path('/data', 'projects', 'gazatents')

image_dir <- file.path(
  '/data/',
  'common',
  'planet',
  'Gaza'
)

out_dir <- file.path(
  '/data',
  'users',
  'doug',
  'wd',
  'realgoodresearch',
  'TentNetFA',
  'hotosm',
  date_tents
)
dir.create(out_dir, showWarnings = F, recursive = T)

#---- load data ----#

# load mastergrid
mastergrid <- rast(file.path(
  data_dir,
  'data',
  'nowpop',
  'mastergrid.tif'
))

# load yellow line exclusion zone
exclusion_zone <- st_read(file.path(
  data_dir,
  'data',
  'ocha',
  'exclusion_zones',
  'buffer_zone_2025-10-10.gpkg'
))

# load tent raster
tent_rast <- rast(file.path(
  data_dir,
  'publications',
  '2026-08-05',
  'supplementary_data',
  paste0('gaza_tent_counts_100m_grid_', gsub('-', '', date_tents), '.tif')
))

# load non-zero strata boundaries
strata_poly <- st_read(
  file.path(
    data_dir,
    'data',
    'ocha',
    'admin',
    'gaza_neighbourhood_boundaries.gpkg'
  )
)

# load zero strata boundaries
strata_zero_poly <- st_read(
  file.path(
    data_dir,
    'data',
    'ocha',
    'admin',
    'gaza_governorate_boundaries.gpkg'
  )
)


#---- update mastergrid ----#

# Remove cells whose centroids fall in an exclusion polygon.
exclusion_zone <- st_transform(exclusion_zone, crs(mastergrid))
mastergrid <- mask(
  mastergrid,
  vect(exclusion_zone),
  inverse = TRUE,
  touches = FALSE
)

# Keep only cell centroids covered by at least one Planet image's alpha band.
study_cells <- which(!is.na(values(mastergrid, mat = FALSE)))
study_centroids <- vect(
  xyFromCell(mastergrid, study_cells),
  type = 'points',
  crs = crs(mastergrid)
)
image_covered <- rep(FALSE, length(study_cells))
for (image_file in file.path(image_dir, planet_images)) {
  alpha <- rast(image_file)[[4]]
  alpha_values <- extract(
    alpha,
    project(study_centroids, crs(alpha)),
    ID = FALSE
  )[[1]]
  image_covered <- image_covered | (!is.na(alpha_values) & alpha_values > 0)
}
mastergrid[study_cells[!image_covered]] <- NA

# Restrict tent counts to the study area on the mastergrid.
stopifnot(compareGeom(tent_rast, mastergrid))
tent_rast <- mask(tent_rast, mastergrid)

# Missing tent counts inside the study area represent zero tents.
tent_rast <- ifel(mastergrid == 1 & is.na(tent_rast), 0, tent_rast)


#---- summary stats for tents ----#

# count of grid cells with tents
sum(tent_rast[!is.na(tent_rast)] > 0)

# sample proportion of total cells
n / sum(tent_rast[!is.na(tent_rast)] > 0)

# summary stats of tent counts
hist(
  tent_rast[!is.na(tent_rast)],
  main = 'Tent count frequencies',
  xlab = 'Tents',
  ylab = '100-m grid cells'
)

quantile(tent_rast[!is.na(tent_rast)], probs = seq(0, 1, by = 0.1))

mean(tent_rast[!is.na(tent_rast)] < 10)
mean(tent_rast[!is.na(tent_rast)] < 30)


#---- design sample ----#

# Sampling frame: positive and zero cells, assigned by cell centroid.
counts <- values(tent_rast, mat = FALSE)
cells <- which(!is.na(counts) & counts >= 0)
frame <- st_as_sf(
  data.frame(cell = cells, count = counts[cells], xyFromCell(tent_rast, cells)),
  coords = c('x', 'y'),
  crs = crs(tent_rast)
)
# Each cell uses the polygon set for its count class; IDs are polygon row numbers.
frame$stratum <- ifelse(frame$count > 0, 'positive', 'zero')
frame$size <- ifelse(frame$count > 0, frame$count, 1)
frame$strata_id <- NA_integer_
n_zero <- round(n * zero_fraction)
stratum_n <- c(positive = n - n_zero, zero = n_zero)
strata_sets <- list(positive = strata_poly, zero = strata_zero_poly)
allocations <- list()
allocation_summaries <- list()

for (s in names(strata_sets)) {
  polygons <- st_transform(strata_sets[[s]], st_crs(frame))
  strata_match <- st_intersects(frame, polygons)
  # A centroid on a shared boundary goes to the first matching polygon.
  strata_id <- vapply(
    strata_match,
    function(x) if (length(x)) x[1] else NA_integer_,
    integer(1)
  )
  in_class <- frame$stratum == s
  frame$strata_id[in_class] <- strata_id[in_class]
  ids <- seq_len(nrow(polygons))
  eligible_cells <- tabulate(strata_id[in_class], nbins = length(ids))
  # For either polygon set, weights are total tents, not zero-cell counts.
  tent_totals <- tapply(frame$count, factor(strata_id, levels = ids), sum)
  tent_totals[is.na(tent_totals)] <- 0
  eligible <- eligible_cells > 0
  weights <- as.numeric(tent_totals)
  weights[!eligible] <- 0

  # Reserve one positive cell per eligible polygon, then apportion the rest.
  allocated <- as.integer(s == 'positive' & eligible)
  remaining <- stratum_n[s] - sum(allocated)
  if (remaining < 0) {
    stop(
      'Positive sample budget is smaller than the number of eligible strata.'
    )
  }
  if (remaining > 0) {
    if (sum(weights) == 0) {
      stop('No tent totals available for this allocation.')
    }
    quota <- remaining * weights / sum(weights)
    extra <- floor(quota)
    remainder <- order(quota - extra, decreasing = TRUE)[seq_len(
      remaining - sum(extra)
    )]
    extra[remainder] <- extra[remainder] + 1L
    allocated <- allocated + extra
  }
  if (any(allocated > eligible_cells)) {
    stop('An allocation exceeds its eligible cell count; review allocation.')
  }

  allocations[[s]] <- data.frame(
    strata_id = ids,
    stratum = s,
    tent_total = as.numeric(tent_totals),
    eligible_cells = eligible_cells,
    n = allocated,
    status = ifelse(
      !eligible,
      'no eligible cells',
      ifelse(allocated == 0, 'zero allocation', 'allocated')
    )
  )
  allocation_summaries[[s]] <- data.frame(
    allocations[[s]],
    st_drop_geometry(polygons),
    check.names = FALSE
  )
  names(allocation_summaries[[s]]) <- make.unique(names(allocation_summaries[[
    s
  ]]))
}
allocation <- do.call(rbind, allocations)
rownames(allocation) <- NULL
print(allocation)
frame <- frame[!is.na(frame$strata_id), ]

# Separate tables because the two designs can use different polygon attributes.
allocation_summary <- allocation_summaries$positive
allocation_zero_summary <- allocation_summaries$zero
write.csv(
  allocation_summary,
  file.path(out_dir, 'sample_allocation.csv'),
  row.names = FALSE
)
write.csv(
  allocation_zero_summary,
  file.path(out_dir, 'sample_allocation_zero.csv'),
  row.names = FALSE
)


#---- draw sample ----#

# Pivotal sampling: PPS for positive counts, equal probabilities for zero counts.
set.seed(as.numeric(gsub('-', '', date_tents)))
frame$inclusion_probability <- 0
frame$selected <- FALSE
for (h in seq_len(nrow(allocation))) {
  i <- which(
    frame$strata_id == allocation$strata_id[h] &
      frame$stratum == allocation$stratum[h]
  )
  if (allocation$n[h] > length(i)) {
    stop('An allocation exceeds its eligible cell count; review allocation.')
  }
  if (allocation$n[h] == 0) {
    next
  }
  pik <- sampling::inclusionprobabilities(frame$size[i], allocation$n[h])
  frame$inclusion_probability[i] <- pik
  frame$selected[i] <- sampling::UPrandompivotal(pik, eps = 1e-10) > 0.5
}

# Sampled cell centroids, with raster cell IDs and inclusion probabilities.
sample_cells <- frame[frame$selected, ]

# Probabilities for ALL eligible cells; excluded cells remain NA.
inclusion_rast <- rast(tent_rast)
values(inclusion_rast) <- NA_real_
inclusion_rast[frame$cell] <- frame$inclusion_probability
names(inclusion_rast) <- 'inclusion_probability'


#---- generate outputs ----#

# Sample mask: 0 for unsampled study-area cells, 1 for sampled cells, NA otherwise.
sample_mask <- rast(tent_rast)
values(sample_mask) <- NA_real_
sample_mask[cells] <- 0
sample_mask[sample_cells$cell] <- 1
names(sample_mask) <- 'sample_mask'

# Tent counts in sampled cells only.
sample_tent_counts <- rast(tent_rast)
values(sample_tent_counts) <- NA_real_
sample_tent_counts[sample_cells$cell] <- sample_cells$count
names(sample_tent_counts) <- 'sample_tent_counts'

# save rasters to disk
writeRaster(
  inclusion_rast,
  file.path(out_dir, 'inclusion_probs.tif'),
  overwrite = TRUE
)
writeRaster(
  sample_mask,
  file.path(out_dir, 'sample_mask.tif'),
  overwrite = TRUE
)
writeRaster(
  sample_tent_counts,
  file.path(out_dir, 'sample_tent_counts.tif'),
  overwrite = TRUE
)
