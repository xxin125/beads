#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace beads {
namespace simulation::output {

class HostWriter {
 public:
  HostWriter();
  ~HostWriter();

  HostWriter(const HostWriter&) = delete;
  HostWriter& operator=(const HostWriter&) = delete;

  void submit(std::function<void()> job);
  void flush();
  std::vector<std::exception_ptr> flush_and_collect_errors();
  void rethrow_if_failed();

 private:
  void worker_loop() noexcept;
  void shutdown_noexcept() noexcept;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> jobs_;
  std::vector<std::exception_ptr> errors_;
  std::thread worker_;
  bool stop_requested_ = false;
  bool worker_active_ = false;
};

}  // namespace simulation::output
}  // namespace beads
