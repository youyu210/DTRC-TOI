#pragma once
#include <algorithm>
#include <cmath>

namespace gipc
{
// One instance per Newton solve. Reference is fixed at the first included
// update, not at the beginning of the warmup. Invalid samples disable only
// this auxiliary exit and never replace the original stopping rule.
struct ToiBudget
{
    bool   active     = false;
    bool   valid      = false;
    bool   pending    = false;
    double reference  = 1;
    double previous   = 0;
    double beta       = 1;
    double budget     = 1;
    double normalized = 0;
    double excess     = 0;
    double alpha      = 0;
    double weight     = 1;
    double unit_budget = 1;

    void start(double residual, double floor, double residual_weight = 1)
    {
        active = true;
        valid = std::isfinite(residual) && residual >= 0 && std::isfinite(floor)
                && floor > 0 && std::isfinite(residual_weight)
                && residual_weight > 0 && residual_weight <= 1;
        weight = residual_weight;
        reference  = valid ? std::max(residual, floor) : 1;
        previous   = residual;
        normalized = residual / reference;
    }

    void accept(double accepted_alpha)
    {
        if(active)
        {
            alpha   = accepted_alpha;
            pending = true;
        }
    }

    void observe(double residual)
    {
        if(!pending)
            return;
        pending = false;
        valid   = valid && std::isfinite(alpha) && alpha >= 0 && alpha <= 1
                && std::isfinite(residual) && residual >= 0;
        if(!valid)
            return;
        const double old  = previous / reference;
        const double next = residual / reference;
        if(!std::isfinite(old) || !std::isfinite(next))
        {
            valid = false;
            return;
        }
        excess = std::max(0.0, next - (1 - alpha) * old);
        beta *= 1 - alpha;
        budget     = (1 - alpha) * budget + weight * excess;
        unit_budget = (1 - alpha) * unit_budget + excess;
        previous   = residual;
        normalized = next;
        valid      = std::isfinite(beta) && std::isfinite(budget)
                     && std::isfinite(unit_budget);
    }
};
}  // namespace gipc
