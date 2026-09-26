#!/usr/bin/env Rscript
# Overlay nearest-neighbour offset histograms for the three pred dates.
# Style: https://r-graph-gallery.com/histogram_several_group.html
# Metrics as grouped bars with values on top:
# https://r-graph-gallery.com/37-barplot-with-number-of-observation.html

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: plot_offset_hist.R <offset_distances.csv> <out.png> [gate_m] [r_library]")
}
csv_path <- args[[1]]
out_png <- args[[2]]
gate_m <- if (length(args) >= 3) as.numeric(args[[3]]) else 5
xlim_m <- 25
date_levels <- c("20251201", "20260220", "20260325")
date_cols <- c("20251201" = "#69b3a2", "20260220" = "#404080", "20260325" = "#e67e22")

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
rlib <- if (length(args) >= 4) args[[4]] else file.path(script_dir, "..", "..", ".Rlibrary")
if (dir.exists(rlib)) {
  .libPaths(c(normalizePath(rlib), .libPaths()))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(grid)
  library(gridExtra)
})

data <- read.csv(csv_path, stringsAsFactors = FALSE)
data$date <- factor(data$date, levels = date_levels)

p_hist <- data %>%
  ggplot(aes(x = distance_m, fill = date)) +
  geom_histogram(color = "#e9ecef", alpha = 0.55, position = "identity", binwidth = 0.5) +
  geom_vline(
    xintercept = gate_m,
    colour = "#c0392b",
    linetype = "dashed",
    linewidth = 0.8
  ) +
  scale_fill_manual(values = date_cols) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  coord_cartesian(xlim = c(0, xlim_m)) +
  labs(
    x = "Nearest UNOSAT tent (m)",
    y = "Count",
    fill = "Pred date",
    title = "v2.2 vs UNOSAT nearest-neighbour offsets",
    subtitle = sprintf(
      "Dashed line = %s m detection gate; bars include the tail beyond it (view 0\u2013%s m)",
      gate_m,
      xlim_m
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

summary_path <- file.path(dirname(csv_path), "summary.csv")
if (!file.exists(summary_path)) {
  stop("Missing summary.csv next to offset_distances.csv: ", summary_path)
}
summary <- read.csv(summary_path, stringsAsFactors = FALSE)
summary <- summary[match(date_levels, summary$name), ]
summary$date <- factor(summary$name, levels = date_levels)

pct_bars <- rbind(
  data.frame(date = summary$date, metric = "Prec", value = 100 * summary$precision,
             label = sprintf("%.1f%%", 100 * summary$precision)),
  data.frame(date = summary$date, metric = "Rec", value = 100 * summary$recall,
             label = sprintf("%.1f%%", 100 * summary$recall)),
  data.frame(date = summary$date, metric = "F1", value = 100 * summary$f1,
             label = sprintf("%.1f%%", 100 * summary$f1)),
  data.frame(
    date = summary$date,
    metric = paste0("NN > ", gate_m, " m"),
    value = 100 * summary$nn_pct_beyond_gate,
    label = sprintf("%.1f%%", 100 * summary$nn_pct_beyond_gate)
  )
)
pct_bars$metric <- factor(
  pct_bars$metric,
  levels = c("Prec", "Rec", "F1", paste0("NN > ", gate_m, " m"))
)

mean_label <- paste0("1-to-1 mean\n(<= ", gate_m, " m)")
metre_bars <- rbind(
  data.frame(date = summary$date, metric = mean_label, value = summary$mean_pos_error_m,
             label = sprintf("%.2f m", summary$mean_pos_error_m)),
  data.frame(date = summary$date, metric = "NN median", value = summary$nn_median_m,
             label = sprintf("%.2f m", summary$nn_median_m))
)
metre_bars$metric <- factor(metre_bars$metric, levels = c(mean_label, "NN median"))

bar_theme <- theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    plot.title = element_text(size = 11),
    plot.caption = element_text(hjust = 0, size = 8, colour = "#2c3e50")
  )

confusion_plot <- function(tp, fp, fn, title) {
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  rec <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  f1 <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
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
    geom_text(aes(label = lab), size = 3, lineheight = 0.95, colour = "#1a1a1a") +
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
      aspect.ratio = 1,
      plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "#2c3e50"),
      panel.grid = element_blank(),
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 8)
    )
}

p_pct <- ggplot(pct_bars, aes(x = metric, y = value, fill = date)) +
  geom_vline(xintercept = c(1.5, 2.5, 3.5), colour = "grey80") +
  geom_col(position = position_dodge(width = 0.85), width = 0.8, alpha = 0.7, colour = "#e9ecef") +
  geom_text(
    aes(label = label),
    position = position_dodge(width = 0.85),
    vjust = -0.8,
    size = 2.8
  ) +
  scale_fill_manual(values = date_cols) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.18)),
    labels = function(x) paste0(x, "%")
  ) +
  coord_cartesian(ylim = c(0, 80), clip = "off") +
  labs(
    x = NULL,
    y = NULL,
    title = "5 m greedy 1-to-1 match",
    caption = sprintf(
      "NN > %s m mean offset: %s",
      gate_m,
      paste(sprintf("%s %.1f m", summary$date, summary$nn_mean_beyond_m), collapse = "  ·  ")
    )
  ) +
  bar_theme

p_m <- ggplot(metre_bars, aes(x = metric, y = value, fill = date)) +
  geom_vline(xintercept = 1.5, colour = "grey80") +
  geom_col(position = position_dodge(width = 0.85), width = 0.8, alpha = 0.7, colour = "#e9ecef") +
  geom_text(
    aes(label = label),
    position = position_dodge(width = 0.85),
    vjust = -0.8,
    size = 2.8
  ) +
  scale_fill_manual(values = date_cols) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  coord_cartesian(ylim = c(0, 6), clip = "off") +
  labs(x = NULL, y = "metres", title = "Matched / nearest offsets") +
  bar_theme

cm_list <- lapply(seq_len(nrow(summary)), function(i) {
  row <- summary[i, ]
  confusion_plot(row$tp, row$fp, row$fn, as.character(row$date))
})
p_cm_all <- confusion_plot(
  sum(summary$tp),
  sum(summary$fp),
  sum(summary$fn),
  "All dates"
)

figure <- arrangeGrob(
  ggplotGrob(p_hist),
  arrangeGrob(ggplotGrob(p_pct), ggplotGrob(p_m), ncol = 2, widths = c(1.15, 1)),
  arrangeGrob(
    ggplotGrob(cm_list[[1]]),
    ggplotGrob(cm_list[[2]]),
    ggplotGrob(cm_list[[3]]),
    ggplotGrob(p_cm_all),
    ncol = 4
  ),
  ncol = 1,
  heights = c(2.2, 1.25, 1.55)
)
ggsave(out_png, figure, width = 12, height = 10.6, dpi = 150, bg = "white")
cat("Wrote", out_png, "\n")
