#include <gipc/utils/toi_budget.h>
#include <cassert>
#include <limits>
#include <iostream>

bool near(double a, double b) { return std::abs(a - b) < 1e-12; }
int main()
{
    gipc::ToiBudget full;
    full.start(10, 1e-30);
    full.accept(1);
    full.observe(2);
    assert(full.valid && near(full.beta, 0) && near(full.budget, .2));
    assert(full.budget > .001); // A full step cannot accept residual=.2.

    gipc::ToiBudget ideal;
    ideal.start(8, 1e-30);
    double residual = 8;
    for(double alpha : {.2, .3, .5, .8})
    {
        ideal.accept(alpha);
        residual *= 1 - alpha;
        ideal.observe(residual);
        assert(near(ideal.excess, 0) && near(ideal.budget, ideal.beta));
        assert(ideal.reference == 8);
    }

    gipc::ToiBudget history;
    history.start(10, 1e-30);
    history.accept(.5);
    history.observe(8);
    assert(near(history.budget, .8));
    history.accept(.5);
    history.observe(1);
    assert(near(history.budget, .4) && near(history.normalized, .1));
    assert(history.budget > history.normalized); // Retained excess history.
    history.accept(0);
    history.observe(30);
    assert(history.budget > 1 && history.budget >= history.normalized);
    assert(history.budget >= history.beta); // Never clamp to [0,1].

    gipc::ToiBudget zero;
    zero.start(0, 1e-30);
    zero.accept(1);
    zero.observe(0);
    assert(zero.valid && zero.reference == 1e-30 && zero.budget == 0);

    for(double alpha : {-1.0, 1.1, std::numeric_limits<double>::quiet_NaN()})
    {
        gipc::ToiBudget invalid;
        invalid.start(10, 1e-30);
        invalid.accept(alpha);
        invalid.observe(1);
        assert(!invalid.valid);
        invalid.accept(.5);
        invalid.observe(0);
        assert(!invalid.valid); // Disabled until next solve, not recovered.
    }
    gipc::ToiBudget invalid_floor;
    invalid_floor.start(1, 0);
    assert(!invalid_floor.valid);
    gipc::ToiBudget invalid_residual;
    invalid_residual.start(1, 1e-30);
    invalid_residual.accept(.5);
    invalid_residual.observe(std::numeric_limits<double>::infinity());
    assert(!invalid_residual.valid);
    for(double weight : {1.0, .1, 1.0/30.0, .01})
    {
        gipc::ToiBudget scaled, unit;
        scaled.start(10,1e-30,weight);
        unit.start(10,1e-30);
        const double samples[] = {9, 12, 2, .02, .1, 0, 3};
        const double alphas[] = {.2, .1, .9, .5, 0, 1, .8};
        for(int i=0;i<7;++i)
        {
            scaled.accept(alphas[i]);unit.accept(alphas[i]);
            scaled.observe(samples[i]);unit.observe(samples[i]);
            assert(scaled.valid);
            assert(scaled.budget+1e-12 >= scaled.beta);
            assert(scaled.budget+1e-12 >= weight*scaled.normalized);
            assert(scaled.budget <= unit.budget+1e-12);
            assert(near(scaled.budget,scaled.beta+weight*(unit.budget-scaled.beta)));
            if(weight==1) assert(scaled.budget==unit.budget);
        }
    }
    gipc::ToiBudget relaxed;
    relaxed.start(1,1e-30,.01);relaxed.accept(1);relaxed.observe(.03);
    assert(relaxed.budget <= .001 && relaxed.normalized <= .1);
    assert(relaxed.unit_budget > .001); // Distinct, explicitly relaxed residual goal.
    gipc::ToiBudget history_gate;
    history_gate.start(1,1e-30,.1);
    history_gate.accept(.99);history_gate.observe(1);
    history_gate.accept(.95);history_gate.observe(.001);
    assert(history_gate.beta <= .001 && history_gate.normalized <= .01);
    assert(history_gate.budget > .001); // Current gate accepts, history still vetoes.
    for(double weight : {0.0,-.1,1.1,std::numeric_limits<double>::infinity()})
    {
        gipc::ToiBudget invalid_weight;invalid_weight.start(1,1e-30,weight);
        assert(!invalid_weight.valid);
    }
    std::cout << "PASS: full-step veto, ideal quadratic decay, history, rebound, fixed reference, zero/floor and invalid inputs\n";
    std::cout << "PASS: dual tolerance bounds, affine identity, weight=1 compatibility and historical veto\n";
}
