#include "host_writer.hpp"

#include <stdexcept>
#include <utility>

namespace beads {
namespace simulation::output {
namespace {

void rethrow_if_present(const std::vector<std::exception_ptr>& errors) {
  if (!errors.empty() && errors.front() != nullptr) {
    std::rethrow_exception(errors.front());
  }
}

}  // namespace

HostWriter::HostWriter()
    : worker_([this]() { worker_loop(); }) {}

HostWriter::~HostWriter() {
  shutdown_noexcept();
}

void HostWriter::submit(std::function<void()> job) {
  if (!job) {
    throw std::invalid_argument("HostWriter requires a valid job.");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    rethrow_if_present(errors_);
    if (stop_requested_) {
      throw std::runtime_error("HostWriter is not accepting new jobs.");
    }
    jobs_.push_back(std::move(job));
  }
  cv_.notify_all();
}

void HostWriter::flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&]() {
    return jobs_.empty() && !worker_active_;
  });
  rethrow_if_present(errors_);
}

std::vector<std::exception_ptr> HostWriter::flush_and_collect_errors() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&]() {
    return jobs_.empty() && !worker_active_;
  });
  return errors_;
}

void HostWriter::rethrow_if_failed() {
  std::lock_guard<std::mutex> lock(mutex_);
  rethrow_if_present(errors_);
}

void HostWriter::worker_loop() noexcept {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&]() {
        return stop_requested_ || !jobs_.empty();
      });
      if (jobs_.empty()) {
        if (stop_requested_) {
          return;
        }
        continue;
      }
      job = std::move(jobs_.front());
      jobs_.pop_front();
      worker_active_ = true;
    }

    try {
      job();
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      errors_.push_back(std::current_exception());
      stop_requested_ = true;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_active_ = false;
    }
    cv_.notify_all();
  }
}

void HostWriter::shutdown_noexcept() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

}  // namespace simulation::output
}  // namespace beads
