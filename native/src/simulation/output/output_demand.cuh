#pragma once

#include <forcefield/force_eval.cuh>
#include <input/native_spec.hpp>

namespace beads {
namespace simulation::output {

// Single source of truth for output-driven force observable demand. Runtime
// output components still own their staging cadence, but ForceEvaluator sizing
// and per-step force observable requests both flow through this plan.
class OutputDemand {
 public:
  static OutputDemand from_spec(const input::OutputSpec& output) noexcept {
    OutputDemand demand;
    if (output.thermo) {
      demand.thermo_enabled_ = true;
      demand.thermo_every_ = output.thermo->every;
      demand.max_force_request_.merge(thermo_force_request());
    }
    return demand;
  }

  const forcefield::ForceEvalRequest& max_force_request() const noexcept {
    return max_force_request_;
  }

  forcefield::ForceEvalRequest force_request(runstep_t step) const noexcept {
    forcefield::ForceEvalRequest request;
    if (thermo_enabled_ &&
        thermo_every_ != 0 &&
        step % thermo_every_ == 0) {
      request.merge(thermo_force_request());
    }
    return request;
  }

 private:
  static forcefield::ForceEvalRequest thermo_force_request() noexcept {
    return forcefield::make_force_eval_request({
        forcefield::ForceObservable::PairPotentialEnergy,
        forcefield::ForceObservable::BondPotentialEnergy,
        forcefield::ForceObservable::AnglePotentialEnergy,
        forcefield::ForceObservable::DihedralPotentialEnergy,
        forcefield::ForceObservable::GlobalScalarVirial,
    });
  }

  bool thermo_enabled_ = false;
  runstep_t thermo_every_ = 0;
  forcefield::ForceEvalRequest max_force_request_;
};

}  // namespace simulation::output
}  // namespace beads
