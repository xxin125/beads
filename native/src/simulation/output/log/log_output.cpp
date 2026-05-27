#include "log_output.cuh"

#include <simulation/output/host_writer.hpp>
#include <simulation/output/thermo/thermo_output.cuh>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace beads {
namespace simulation::output::log {
namespace {

constexpr int kThermoColumnWidth = 16;
constexpr int kLogLabelWidth = 34;
constexpr int kLogValueColumn = 2 + kLogLabelWidth + 3;

std::string log_path_from_prefix(const std::string& prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("output.log.prefix must not be empty.");
  }
  return prefix + ".log";
}

bool echo_includes_screen(const std::string& echo) noexcept {
  return echo == "screen" || echo == "both";
}

bool echo_includes_file(const std::string& echo) noexcept {
  return echo == "log" || echo == "both";
}

std::string enabled_text(bool enabled) {
  return enabled ? "enabled" : "disabled";
}

std::string style_or_none(const std::optional<std::string>& style) {
  return style ? *style : "none";
}

std::string index_or_none(const std::optional<index_t>& value) {
  if (!value) {
    return "none";
  }
  std::ostringstream text;
  text << *value;
  return text.str();
}

std::string format_real3(const std::array<real_t, 3>& value) {
  std::ostringstream text;
  text << '(' << value[0] << ',' << value[1] << ',' << value[2] << ')';
  return text.str();
}

std::string format_log_label_value(
    std::string_view label,
    const std::string& value,
    int indent = 2,
    int value_column = kLogValueColumn) {
  const std::string prefix =
      std::string(static_cast<std::size_t>(indent), ' ') +
      std::string(label);
  std::ostringstream line;
  line << prefix;
  constexpr int separator_width = 3;
  if (value_column > static_cast<int>(prefix.size()) + separator_width) {
    line << std::string(
        static_cast<std::size_t>(
            value_column - static_cast<int>(prefix.size()) - separator_width),
        ' ');
  } else {
    line << ' ';
  }
  line << ": " << value;
  return line.str();
}

template <typename T>
std::string format_log_label_value(
    std::string_view label,
    const T& value,
    int indent = 2,
    int value_column = kLogValueColumn) {
  std::ostringstream value_stream;
  value_stream << value;
  return format_log_label_value(
      label,
      value_stream.str(),
      indent,
      value_column);
}

std::string format_thermo_header_line(const thermo::ThermoLogRow& row) {
  std::ostringstream line;
  line << std::setw(kThermoColumnWidth) << "Step"
       << std::setw(kThermoColumnWidth) << "PotEng"
       << std::setw(kThermoColumnWidth) << "PairEng";
  if (row.schema.include_bond_pe) {
    line << std::setw(kThermoColumnWidth) << "BondEng";
  }
  if (row.schema.include_angle_pe) {
    line << std::setw(kThermoColumnWidth) << "AngleEng";
  }
  if (row.schema.include_dihedral_pe) {
    line << std::setw(kThermoColumnWidth) << "DihedEng";
  }
  line << std::setw(kThermoColumnWidth) << "KinEng"
       << std::setw(kThermoColumnWidth) << "Temp"
       << std::setw(kThermoColumnWidth) << "Press";
  return line.str();
}

std::string format_thermo_row_line(const thermo::ThermoLogRow& row) {
  std::ostringstream line;
  const auto real_text = [](real_t value) {
    std::ostringstream text;
    text << std::setprecision(9) << value;
    return text.str();
  };
  line << std::setw(kThermoColumnWidth) << row.step
       << std::setw(kThermoColumnWidth) << real_text(row.pe)
       << std::setw(kThermoColumnWidth) << real_text(row.pair_pe);
  if (row.schema.include_bond_pe) {
    line << std::setw(kThermoColumnWidth) << real_text(row.bond_pe);
  }
  if (row.schema.include_angle_pe) {
    line << std::setw(kThermoColumnWidth) << real_text(row.angle_pe);
  }
  if (row.schema.include_dihedral_pe) {
    line << std::setw(kThermoColumnWidth) << real_text(row.dihedral_pe);
  }
  line << std::setw(kThermoColumnWidth) << real_text(row.ke)
       << std::setw(kThermoColumnWidth) << real_text(row.temp)
       << std::setw(kThermoColumnWidth) << real_text(row.press);
  return line.str();
}

}  // namespace

class LogOutput::Impl {
 public:
  Impl(LogOutputConfig config, HostWriter& writer)
      : echo_(std::move(config.echo)),
        writer_(writer) {
    if (echo_ != "screen" && echo_ != "log" &&
        echo_ != "both" && echo_ != "none") {
      throw std::invalid_argument(
          "output.log.echo must be screen, log, both, or none.");
    }
    write_screen_ = echo_includes_screen(echo_);
    write_file_ = echo_includes_file(echo_);
    if (write_file_) {
      if (!config.prefix) {
        throw std::invalid_argument(
            "output.log.prefix must be provided for file logging.");
      }
      path_ = log_path_from_prefix(*config.prefix);
    } else if (config.prefix) {
      throw std::invalid_argument(
          "output.log.prefix is only supported for log or both echo.");
    }
  }

  ~Impl() {
    try {
      writer_.flush();
      flush_file();
    } catch (...) {
    }
  }

  bool active() const noexcept {
    return write_screen_ || write_file_;
  }

  const std::string& echo() const noexcept {
    return echo_;
  }

  void submit_run_start(LogRunStartSummary summary) {
    if (!active()) {
      return;
    }
    writer_.submit([this, summary = std::move(summary)]() {
      write_run_start(summary);
    });
  }

  void write_thermo_row_from_writer(const thermo::ThermoLogRow& row) {
    if (!active()) {
      return;
    }
    write_thermo_header_once(row);
    write_message("THERMO", format_thermo_row_line(row));
  }

  void submit_run_end(LogRunEndSummary summary) {
    if (!active()) {
      return;
    }
    writer_.submit([this, summary]() {
      write_run_end(summary);
    });
  }

  void flush_file() {
    if (stream_.is_open()) {
      stream_.flush();
      if (!stream_) {
        throw std::runtime_error("failed to flush log output file " + path_ + ".");
      }
    }
    if (write_screen_) {
      std::cout.flush();
    }
  }

 private:
  void write_message(std::string_view level, const std::string& message) {
    const bool separate_before = wrote_any_ && level != "THERMO";
    const std::string line =
        std::string("[BEADS] ") + std::string(level) + " " + message;
    if (write_screen_) {
      if (separate_before) {
        std::cout << '\n';
      }
      std::cout << line << '\n';
      if (!std::cout) {
        throw std::runtime_error("failed to write log output to screen.");
      }
    }
    if (write_file_) {
      ensure_file_open();
      if (separate_before) {
        stream_ << '\n';
      }
      stream_ << line << '\n';
      if (!stream_) {
        throw std::runtime_error("failed to write log output file " + path_ + ".");
      }
    }
    wrote_any_ = true;
  }

  void write_run_start(const LogRunStartSummary& summary) {
    {
      std::ostringstream message;
      message << "execute start"
              << '\n' << format_log_label_value("units", summary.units)
              << '\n' << format_log_label_value("particles", summary.n_particles)
              << '\n' << format_log_label_value("box_lo", format_real3(summary.box_lower))
              << '\n' << format_log_label_value("box_hi", format_real3(summary.box_upper))
              << '\n' << format_log_label_value(
                  "box_lengths",
                  format_real3(summary.box_lengths))
              << '\n' << format_log_label_value("runsteps", summary.runsteps);
      write_message("INFO", message.str());
    }
    {
      std::ostringstream message;
      message << "dynamics summary"
              << '\n' << format_log_label_value("style", summary.dynamics_style)
              << '\n' << format_log_label_value(
                  "thermostat",
                  summary.thermostat_style);
      write_message("INFO", message.str());
    }
    {
      std::ostringstream message;
      message << "neighbor summary"
              << '\n' << format_log_label_value(
                  "cutoff_buffer",
                  summary.cutoff_buffer)
              << '\n' << format_log_label_value("path", summary.neighbor_path)
              << '\n' << format_log_label_value(
                  "rebuild_check_every",
                  summary.rebuild_check_every)
              << '\n' << format_log_label_value(
                  "sort_every_rebuild",
                  summary.sort_every_rebuild)
              << '\n' << format_log_label_value("max_neighbors", summary.max_neighbors);
      write_message("INFO", message.str());
    }
    {
      std::ostringstream message;
      message << "forcefield summary"
              << '\n' << format_log_label_value("pair", summary.pair_style)
              << '\n' << format_log_label_value(
                  "bond",
                  style_or_none(summary.bond_style))
              << '\n' << format_log_label_value(
                  "angle",
                  style_or_none(summary.angle_style))
              << '\n' << format_log_label_value(
                  "dihedral",
                  style_or_none(summary.dihedral_style))
              << '\n' << format_log_label_value(
                  "bonded_exclusion_policy",
                  summary.bonded_exclusion_policy)
              << '\n' << format_log_label_value(
                  "bonded_exclusion_distance",
                  index_or_none(summary.bonded_exclusion_distance))
              << '\n' << format_log_label_value(
                  "bonded_excluded_pairs",
                  summary.bonded_excluded_pair_count);
      write_message("INFO", message.str());
    }
    {
      std::ostringstream message;
      message << "topology summary"
              << '\n' << format_log_label_value("bonds", summary.bond_count)
              << '\n' << format_log_label_value("angles", summary.angle_count)
              << '\n' << format_log_label_value("dihedrals", summary.dihedral_count);
      write_message("INFO", message.str());
    }
    {
      std::ostringstream message;
      message << "output summary"
              << '\n' << format_log_label_value(
                  "thermo",
                  enabled_text(summary.thermo_enabled))
              << '\n' << format_log_label_value(
                  "final_state",
                  enabled_text(summary.final_state_enabled))
              << '\n' << format_log_label_value(
                  "trajectory",
                  enabled_text(summary.trajectory_enabled))
              << '\n' << format_log_label_value("log", summary.log_echo);
      write_message("INFO", message.str());
    }
  }

  void ensure_file_open() {
    if (stream_.is_open()) {
      return;
    }
    stream_.open(path_, std::ios::out | std::ios::trunc);
    if (!stream_) {
      throw std::runtime_error(
          "failed to open log output file " + path_ + ".");
    }
  }

  void write_thermo_header_once(const thermo::ThermoLogRow& row) {
    if (thermo_header_written_) {
      return;
    }
    write_message("THERMO", format_thermo_header_line(row));
    thermo_header_written_ = true;
  }

  void write_run_end(const LogRunEndSummary& summary) {
    std::ostringstream message;
    message << "execute end"
            << '\n' << format_log_label_value("runsteps", summary.runsteps);
    {
      std::ostringstream value;
      value << std::setprecision(9) << summary.run_wall_s;
      message << '\n' << format_log_label_value("run_wall_s", value.str());
    }
    if (summary.steps_per_second) {
      std::ostringstream value;
      value << std::setprecision(9) << *summary.steps_per_second;
      message << '\n' << format_log_label_value(
          "steps_per_second",
          value.str());
    } else {
      message << '\n' << format_log_label_value(
          "steps_per_second",
          "not_applicable");
    }
    message << '\n' << format_log_label_value(
        "neighbor_build_count",
        summary.neighbor_build_count)
            << '\n' << format_log_label_value(
        "dangerous_rebuild_count",
        summary.dangerous_rebuild_count);
    write_message("INFO", message.str());
  }

  std::string echo_;
  std::string path_;
  HostWriter& writer_;
  std::ofstream stream_;
  bool write_screen_ = false;
  bool write_file_ = false;
  bool thermo_header_written_ = false;
  bool wrote_any_ = false;
};

LogOutput::LogOutput(LogOutputConfig config, HostWriter& writer)
    : impl_(std::make_unique<Impl>(std::move(config), writer)) {}

LogOutput::~LogOutput() = default;

bool LogOutput::active() const noexcept {
  return impl_->active();
}

const std::string& LogOutput::echo() const noexcept {
  return impl_->echo();
}

void LogOutput::submit_run_start(LogRunStartSummary summary) {
  impl_->submit_run_start(std::move(summary));
}

void LogOutput::write_thermo_row_from_writer(
    const thermo::ThermoLogRow& row) {
  impl_->write_thermo_row_from_writer(row);
}

void LogOutput::submit_run_end(LogRunEndSummary summary) {
  impl_->submit_run_end(summary);
}

void LogOutput::flush_file() {
  impl_->flush_file();
}

}  // namespace simulation::output::log
}  // namespace beads
