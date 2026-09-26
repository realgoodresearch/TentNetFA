#!/usr/bin/env Rscript
# One holdout fold, same layout as point_match/plot_offset_hist.R.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: plot_offset_hist.R <offset_distances.csv> <summary.csv> <out.png> [gate_m] [r_library]")
}
csv_path <- args[[1]]
summary_path <- args[[2]]
out_png <- args[[3]]
gate_m <- if (length(args) >= 4) as.numeric(args[[4]]) else 5

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
rlib <- if (length(args) >= 5) args[[5]] else file.path(script_dir, "..", "..", ".Rlibrary")
if (dir.exists(rlib)) {
  .libPaths(c(normalizePath(rlib), .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(grid)
  library(gridExtra)
})

fill_col <- "#1a5276"
data <- read.csv(csv_path, stringsAsFactors = FALSE)
summary <- read.csv(summary_path, stringsAsFactors = FALSE)
if (nrow(summary) != 1) {
  stop("summary.csv must have one row")
}
row <- summary[1, ]
fold_label <- as.character(row$name)

p_hist <- data %>%
  ggplot(aes(x = distance_m)) +
  geom_histogram(fill = fill_col, color = "#e9ecef", alpha = 0.75, binwidth = 0.5) +
  geom_vline(
    xintercept = gate_m,
    colour = "#c0392b",
    linetype = "dashed",
    linewidth = 0.8
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(xlim = c(0, 25)) +
  labs(
    x = "Nearest snapshot tent (m)",
    y = "Count",
    title = paste0(fold_label, " nearest-neighbour offsets"),
    subtitle = sprintf(
      "Dashed line = %s m detection gate; bars include the tail beyond it (view 0\u201325 m)",
      gate_m
    )
  ) +
  theme_minimal(base_size = 12)

pct_bars <- data.frame(
  metric = c("Prec", "Rec", "F1", paste0("NN > ", gate_m, " m")),
  value = c(
    100 * row$precision,
    100 * row$recall,
    100 * row$f1,
    100 * row$nn_pct_beyond_gate
  ),
  label = sprintf(
    "%.1f%%",
    c(100 * row$precision, 100 * row$recall, 100 * row$f1, 100 * row$nn_pct_beyond_gate)
  )
)
pct_bars$metric <- factor(pct_bars$metric, levels = pct_bars$metric)

mean_label <- paste0("1-to-1 mean\n(<= ", gate_m, " m)")
metre_bars <- data.frame(
  metric = c(mean_label, "NN median"),
  value = c(row$mean_pos_error_m, row$nn_median_m),
  label = sprintf("%.2f m", c(row$mean_pos_error_m, row$nn_median_m))
)
metre_bars$metric <- factor(metre_bars$metric, levels = metre_bars$metric)

bar_theme <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.x = element_blank(),
    plot.title = element_text(size = 11),
    plot.caption = element_text(hjust = 0, size = 8, colour = "#2c3e50")
  )

pct_top <- max(80, max(pct_bars$value) * 1.18)
p_pct <- ggplot(pct_bars, aes(x = metric, y = value)) +
  geom_vline(xintercept = c(1.5, 2.5, 3.5), colour = "grey80") +
  geom_col(fill = fill_col, width = 0.72, alpha = 0.8, colour = "#e9ecef") +
  geom_text(aes(label = label), vjust = -0.6, size = 3.1) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.08)),
    labels = function(x) paste0(x, "%")
  ) +
  coord_cartesian(ylim = c(0, pct_top), clip = "off") +
  labs(
    x = NULL,
    y = NULL,
    title = sprintf("%s m greedy 1-to-1 match", gate_m),
    caption = sprintf("NN > %s m mean offset: %.1f m", gate_m, row$nn_mean_beyond_m)
  ) +
  bar_theme

metre_top <- max(6, max(metre_bars$value) * 1.25)
p_m <- ggplot(metre_bars, aes(x = metric, y = value)) +
  geom_vline(xintercept = 1.5, colour = "grey80") +
  geom_col(fill = fill_col, width = 0.72, alpha = 0.8, colour = "#e9ecef") +
  geom_text(aes(label = label), vjust = -0.6, size = 3.1) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(ylim = c(0, metre_top), clip = "off") +
  labs(x = NULL, y = "metres", title = "Matched / nearest offsets") +
  bar_theme

confusion_plot <- function(tp, fp, fn, title, prec, rec, f1) {
  cells <- data.frame(
    actual = factor(
      c("Positive", "Positive", "Negative", "Negative"),
      levels = c("Positive", "Negative")
    ),
    predicted = factor(
      c("Positive", "Negative", "Positive", "Negative"),
      levels = c("Positive", "Negative")
    ),
    n = c(tp, fn, fp, NA_real_),
    lab = c(
      sprintf("TP\n%s", format(as.integer(tp), big.mark = ",", scientific = FALSE)),
      sprintf("FN\n%s", format(as.integer(fn), big.mark = ",", scientific = FALSE)),
      sprintf("FP\n%s", format(as.integer(fp), big.mark = ",", scientific = FALSE)),
      "TN\nn/a"
    )
  )
  ggplot(cells, aes(x = predicted, y = actual, fill = n)) +
    geom_tile(colour = "white", linewidth = 1.4) +
    geom_text(aes(label = lab), size = 3.2, lineheight = 0.95, colour = "#1a1a1a") +
    scale_fill_gradient(low = "#d4e6f1", high = "#1a5276", na.value = "#eeeeee", guide = "none") +
    scale_x_discrete(position = "top") +
    scale_y_discrete(limits = rev) +
    labs(
      x = "Predicted",
      y = "Actual",
      title = title,
      subtitle = sprintf("P %.1f%%   R %.1f%%   F1 %.1f%%", 100 * prec, 100 * rec, 100 * f1)
    ) +
    theme_minimal(base_size = 9) +
    theme(
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#2c3e50"),
      panel.grid = element_blank(),
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 8),
      plot.margin = margin(4, 4, 4, 4)
    )
}

p_cm <- confusion_plot(
  row$tp, row$fp, row$fn, sub(" holdout vs snapshot", "", fold_label),
  row$precision, row$recall, row$f1
)

figure <- arrangeGrob(
  ggplotGrob(p_hist),
  arrangeGrob(
    ggplotGrob(p_pct),
    ggplotGrob(p_m),
    ggplotGrob(p_cm),
    ncol = 3,
    widths = c(1.25, 0.95, 1.05)
  ),
  ncol = 1,
  heights = c(1.35, 1)
)
ggsave(out_png, figure, width = 12.2, height = 7.2, dpi = 150, bg = "white")
cat("Wrote", out_png, "\n")
