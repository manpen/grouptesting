#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include <tlx/die.hpp>
#include <tlx/logger.hpp>
#include <tlx/algorithm.hpp>

#include <range/v3/action.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/view.hpp>

using pat_id = uint32_t;
using test_id = uint32_t;

enum class InferenceAlgo {
    Comp, DD
};

struct TestDesign {
    template <typename T>
    class RangeAdapter {
    public:
        RangeAdapter(T begin, T end) : begin_(begin), end_(end) {}

        T begin() {return begin_;}
        T end() {return end_;}

    private:
        T begin_;
        T end_;
    };

    auto tests_of_patient(pat_id u) {
        assert(u < num_patients_);
        return RangeAdapter{tests_indices_[u], tests_indices_[u+1]};
    }

    auto tests_of_patient(pat_id u) const {
        assert(u < num_patients_);
        return RangeAdapter{tests_indices_[u], tests_indices_[u+1]};
    }

    constexpr static bool debug = false;
    using hash_set = std::unordered_set<pat_id>;

    TestDesign() = default;

    void generate_design_delta(pat_id num_patients, test_id num_tests, test_id tests_per_patient, std::mt19937_64& gen) {
        num_patients_ = num_patients;
        num_tests_ = num_tests;
        tests_per_patient_ = tests_per_patient;
        num_edges_ = num_patients * tests_per_patient;
        patients_per_tests_ = num_edges_ / num_tests_;

        tests_.resize(num_edges_);
        tests_indices_.resize(num_patients_ + 1);

        tests_indices_.front() = tests_.begin();

        std::uniform_int_distribution<test_id> distr(0, num_tests_ - 1);
        std::unordered_set<test_id> tests_of_patient;
        tests_of_patient.reserve(2*tests_per_patient);
        for(pat_id p = 0; p < num_patients; p++) {
            tests_of_patient.clear();
            while(tests_of_patient.size() < tests_per_patient)
                tests_of_patient.insert(distr(gen));

            auto begin = tests_.begin() + (tests_per_patient * p);
            auto end = begin + tests_per_patient;
            tests_indices_[p+1] = end;
            std::copy(tests_of_patient.cbegin(), tests_of_patient.cend(), begin);
            std::sort(begin, end);
        }
    }

    void generate_design_gamma(pat_id num_patients, test_id num_tests, double patients_per_test, std::mt19937_64& gen, bool sparse = false) {
        // Setup model
        num_patients_ = num_patients;
        num_tests_ = num_tests;
        patients_per_tests_ = patients_per_test;
        num_edges_ = std::ceil(num_tests * patients_per_test);
        tests_per_patient_ = num_edges_ / num_patients;

        tests_.resize(num_edges_);
        tests_indices_.resize(num_patients_ + 1);

        // create ~patients_per_test balls per test and obtain a
        // random permutation of the balls
        {
            auto edges_per_test = num_edges_ / num_tests_;
            auto tests_with_extra = num_edges_ % num_tests_;

            auto it = tests_.begin();
            for(test_id u = 0; u < num_tests_; ++u) {
                const auto n = edges_per_test + (u < tests_with_extra);
                std::fill_n(it, n, u);
                it += n;
            }

            assert(it == tests_.end());

            std::shuffle(tests_.begin(), tests_.end(), gen);
        }

        auto edges_per_patient = num_edges_ / num_patients_;
        auto patients_with_extra = num_edges_ % num_patients_;
        auto patients_wo_extra = num_patients_ - patients_with_extra;

        // make sure singles do not collide
        if (sparse && edges_per_patient == 1 && patients_wo_extra > 1) {
            tlx_die_unless(patients_wo_extra <= num_tests);
            std::set<pat_id> singles;

            size_t num_swaps = 0;
            for(size_t i = 0; i < patients_wo_extra; ++i) {
                while (!singles.insert(tests_[i]).second) {
                    std::uniform_int_distribution<size_t> distr(i+1, tests_.size() - 1);
                    std::swap(tests_[i], tests_[distr(gen)]);
                    num_swaps++;
                }
            }

            std::cout << "Swaps: " << num_swaps << "\n";
        }

        // draw ~tests_per_patient_ balls per patient and remove duplicates
        {
            auto read = tests_.begin();
            auto write = tests_.begin();
            tests_indices_.front() = write;

            for (pat_id u = 0; u < num_patients_; ++u) {
                const auto take = edges_per_patient + (u >= patients_wo_extra);

                if (read != write) std::move(read, read + take, write);

                std::sort(write, write + take);

                write = std::unique(write, write + take);
                tests_indices_[u + 1] = write;
                read += take;
            }
            assert(read == tests_.end());

            tests_.erase(write, tests_.end());
            LOG << "number of duplicates: " << (num_edges_ - tests_.size());
        }
    }

    void cough(double theta, std::mt19937_64& gen) {
        theta_ = theta;
        const size_t num_infected = std::pow(num_patients_, theta);

        infected_patients_.clear();
        infected_patients_.reserve(2 * num_infected);
        positive_tests_.clear();
        positive_tests_.reserve(std::min<size_t>(num_tests_, num_infected * tests_per_patient()));
        positive_tests_bitmap_.resize(num_tests_);
        std::fill(positive_tests_bitmap_.begin(), positive_tests_bitmap_.end(), 0);

        std::uniform_int_distribution<pat_id> distr(0, num_patients_ - 1);
        while (infected_patients_.size() < num_infected) {
            auto patient_idx = distr(gen);
            if (!infected_patients_.insert(patient_idx).second)
                continue; // already infected, cough again

            for (auto t : tests_of_patient(patient_idx)) {
                if (positive_tests_bitmap_[t]) continue;
                positive_tests_.insert(t);
                positive_tests_bitmap_[t] = true;
            }
        }

        LOG << "infected: " << infected_patients_.size();
        LOG << "positive tests: " << positive_tests_.size() << " (" << (100.0 * positive_tests_.size() / num_tests_) << "%)";
        LOG << "spread factor: " << (1.0 * positive_tests_.size() / infected_patients_.size());
    }

    hash_set algo_comp() const {
        hash_set infected;

        // find set of patients with at least one positive test
        for(pat_id p = 0; p < num_patients_; ++p) {
            if (ranges::any_of(tests_of_patient(p),
                               [&] (test_id t) {return !positive_tests_bitmap_[t];}))
                continue;

            infected.insert(p);
        }

        return infected;
    }

    std::pair<hash_set, hash_set> algos_comp_and_dd() const {
        auto candidates = algo_comp();
        //std::unordered_map<test_id, pat_id> counts;
        std::vector<pat_id> counts(num_tests());

        for(pat_id p : candidates) {
            for(test_id t : tests_of_patient(p)) {
                counts[t]++;
            }
        }


        hash_set surely_infected;
        for(pat_id p : candidates) {
            for(test_id t : tests_of_patient(p)) {
                if (ranges::any_of(tests_of_patient(p),
                                   [&] (test_id t) {return counts[t] == 1;}))
                    surely_infected.insert(p);
            }
        }

        return {candidates, surely_infected};
    }

    hash_set algo_dd() const {
        return algos_comp_and_dd().second;
    }

    hash_set algo(InferenceAlgo algo) const {
        return (algo == InferenceAlgo::Comp) ? algo_comp() : algo_dd();
    }

    size_t num_patients()       const {return num_patients_;}
    size_t num_tests()          const {return num_tests_;}
    double tests_per_patient()  const {return tests_per_patient_;}
    size_t num_edges()          const {return num_edges_;}
    double patients_per_tests() const {return patients_per_tests_;}
    double theta()              const {return theta_; }

    size_t num_single_edges()   const {return std::distance(tests_indices_.front(), tests_indices_.back());}

    const hash_set& infected_patients() const {return infected_patients_;}
    const std::unordered_set<test_id>& positive_tests() const {return positive_tests_;}

private:
    size_t num_patients_;
    size_t num_tests_;
    double tests_per_patient_;
    size_t num_edges_;
    double patients_per_tests_;
    double theta_{-1.0};

    std::vector<test_id>  tests_;
    std::vector<std::vector<test_id>::iterator> tests_indices_;

    hash_set infected_patients_;
    std::unordered_set<test_id> positive_tests_;
    std::vector<bool> positive_tests_bitmap_;
};
