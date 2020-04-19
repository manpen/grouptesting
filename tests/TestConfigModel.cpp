#include <iostream>
#include <unordered_set>

#include <tlx/die.hpp>
#include <TestDesign.hpp>

constexpr bool debug = 1;

int main() {
    std::mt19937_64 gen(1);
    TestDesign td;

    size_t num_collisions = 0;
    size_t num_repeats = 1;
    for(unsigned repeat = 0; repeat < num_repeats; ++repeat) {
        size_t num_patients = 999;
        size_t Gamma = 3;
        size_t num_tests = num_patients / Gamma;
        double theta = 0.3;

        td.generate_design_gamma(num_patients, num_tests, Gamma, gen);

        // check config model
        for (size_t p = 0; p < num_patients; ++p) {
            auto r = td.tests_of_patient(p);
            die_unless(std::distance(r.begin(), r.end()) > 0);
        }

        // check spread
        td.cough(theta, gen);
        die_unless(td.infected_patients().size() >= static_cast<size_t>(std::pow(num_patients, theta)));
        die_unless(td.infected_patients().size() <= static_cast<size_t>(std::pow(num_patients, theta)) + 1);

        LOG << "Infected";
        std::unordered_set<test_id> positive_tests;
        bool collision = false;
        for(auto inf : td.infected_patients()) {
            LOG << "  Patient " << inf;
            auto r = td.tests_of_patient(inf);
            for(auto test : r) {
                LOG << "    Test: " << test;
                auto res = positive_tests.insert(test);
                collision |= !res.second;
            }
        }

        num_collisions += collision;
        if (collision) continue;

        const auto dd   = td.algo_dd();
        const auto comp = td.algo_comp();

        LOG << "Comp";
        for(auto x : comp) {
            LOG << "  Found: " << x;
            auto r = td.tests_of_patient(x);
            const auto in_pos_test = std::any_of(r.begin(), r.end(), [&] (test_id t) {
                return positive_tests.find(t) != positive_tests.end();
            });
            die_unless(in_pos_test);
        }

        LOG << "  Check all infected were found";
        for(auto p : td.infected_patients()) {
            LOG << "    " << p;
            die_unless(comp.find(p) != comp.end());
        }

        LOG << "DD";
        for(auto x : dd) {
            LOG << "  Found: " << x;
        }

            //die_unless(td.algo_dd() == td.infected_patients());
    }

    LOG1 << "Num collisions: " << num_collisions << " Rate: " << (static_cast<double>(num_collisions) / num_repeats);

    return 0;
}