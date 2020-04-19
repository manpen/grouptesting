#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include <ConfusionMatrix.hpp>
#include <TempFile.hpp>
#include <TestDesign.hpp>

#include <tlx/math.hpp>

#include <omp.h>

std::pair<unsigned, std::mt19937_64> get_prng() {
    static std::mt19937_64 seed_seq(std::random_device{}());
    auto seed = std::uniform_int_distribution<unsigned>{}(seed_seq);
    auto gen = std::mt19937_64{seed};
    return {seed, gen};
}

void print_sim_header(std::ostream &os) {
    os << "seed,"
          "n,"
          "m,"
          "Gamma,"
          "Delta,"
          "theta,"
          "numEdges,"
          "numSingleEdges,"
          "k,"
          "algo,"
          "series,"
          "tp,tn,fp,fn,"
          "acc\n";
}

void report(std::ostream &os, const TestDesign &td, unsigned seed, const TestDesign::hash_set &infs,
            InferenceAlgo algo, const std::string &series = "") {
    auto conf = ConfusionMatrix::compute(td.infected_patients(), infs, td.num_patients());
    tlx_die_unless(algo == InferenceAlgo::DD || conf.false_negative() == 0);
    tlx_die_unless(algo == InferenceAlgo::Comp || conf.false_positive() == 0);

    std::stringstream ss;
    ss << seed << ',' << td.num_patients() << ',' << td.num_tests() << ','
       << td.patients_per_tests() << ',' << td.tests_per_patient() << ',' << td.theta() << ','
       << td.num_edges() << ',' << td.num_single_edges() << ',' << td.infected_patients().size()
       << ',' << (algo == InferenceAlgo::DD ? "dd" : "comp") << ',' << series << ','
       << conf.true_positive() << ',' << conf.true_negative() << ',' << conf.false_positive() << ','
       << conf.false_negative() << ',' << conf.accuracy() << '\n';

    os << ss.str();
}

void simulate_and_report(std::ostream &os, const TestDesign &td, unsigned seed) {
    auto inferred = td.algos_comp_and_dd();
    report(os, td, seed, inferred.first, InferenceAlgo::Comp);
    report(os, td, seed, inferred.second, InferenceAlgo::DD);
}


template <typename Func>
size_t bin_search(size_t left, size_t right, Func f, bool negateFunction = false) {
    while (right - left > 1) {
        auto middle = left + (right - left) / 2;

        if (f(middle) != negateFunction)
            right = middle;
        else
            left = middle;
    }

    return left;
}

template <typename Func>
size_t exp_search(size_t n0, Func f, bool negateFunction = false) {
    // grow
    size_t step = 1;
    size_t stepprev;
    while (true) {
        stepprev = step;
        step = stepprev * 2;
        if (f(n0 + step) != negateFunction)
            break;
    }

    // shrink
    return bin_search(n0 + stepprev, n0 + step, f, negateFunction);
}

template <typename Func>
double decide_accuracy(Func comp, bool lower) {
    constexpr unsigned num_samples = 10;
    constexpr double extreme_threshold = 0.1;
    constexpr unsigned slack = 1;

    // is false if whp inference fails
    int violations = 0;
    for (auto r = 0; r < num_samples; ++r) {
        const auto acc = comp();

        violations += lower ? (acc > extreme_threshold) : (acc < 1.0 - extreme_threshold);

        if (violations > slack)
            return true;
    }

    return false;
}

auto &get_threadlocal_prng() {
    thread_local static std::mt19937_64 gen(std::random_device{}());
    return gen;
}

void print_search_header(std::ostream& out) {
    out << "seed,"
                  "n,"
                  "Gamma,"
                  "theta,"
                  "algo,"
                  "m_search_min,"
                  "m_search_max,"
                  "m_lower,"
                  "m_upper\n";
}

void search_const_test(std::ostream& out, InferenceAlgo algo, size_t n) {
    const auto num_steps = 30;

    for (double Gamma : {3., 5.}) {
        const auto max_k = static_cast<size_t>(std::pow(n, Gamma / (Gamma + 1)));
        for(unsigned k = 2; k < max_k; k += (max_k + num_steps - 1) / num_steps) {
            const auto theta = std::log(k) / std::log(n);
            std::cout << (algo == InferenceAlgo::Comp ? "COMP" : "DD") << ' ' << "n=" << n << ' '
                      << "Gamma=" << Gamma << ' ' << "theta=" << theta << ' ' << "k=" << k
                      << std::endl;
            if (theta > Gamma / (Gamma + 1)) {
                std::cout << "      Skip as theta > Gamma / (Gamma + 1)" << std::endl;
                continue;
            }

            auto m_min = static_cast<size_t>(n / Gamma);
            auto m_max = 4 * n;

            size_t found_valid = 0;

            #pragma omp parallel reduction(+:found_valid)
            {
                auto &gen = get_threadlocal_prng();
                auto seed = 0;
                TestDesign td;
                std::stringstream ss;

                const auto nthreads = omp_get_thread_num();
                #pragma omp parallel for
                for (int repeat = 0; repeat < static_cast<int>(4.8 * nthreads); ++repeat) {
                    auto check_stable_state = [&](size_t m, bool check_is_correct) {
                        unsigned violations = 0;
                        for (unsigned r = 0; r < 10; r++) {
                            td.generate_design_gamma(n, m, Gamma, gen);
                            td.cough(theta, gen);

                            auto res = td.algo(algo);
                            auto cm = ConfusionMatrix::compute(td.infected_patients(), res, n);

                            if (check_is_correct)
                                violations += (cm.size_false() > 0);
                            else
                                violations += (cm.size_false() == 0);

                            if (violations >= 1)
                                return false;
                        };

                        return true;
                    };

                    auto is_incorrect_whp = [&](size_t m) { return check_stable_state(m, false); };
                    auto is_correct_whp = [&](size_t m) { return check_stable_state(m, true); };

                    int64_t m_upper = -1;
                    if (is_correct_whp(m_max)) {
                        m_upper = bin_search(m_min, m_max, is_correct_whp);
                    }

                    int64_t m_lower = -1;
                    if (is_incorrect_whp(m_min) && (m_upper > 0 || !is_incorrect_whp(m_max))) {
                        m_lower = bin_search(m_min, std::max<int64_t>(m_upper, m_max), is_incorrect_whp, true);
                    }

                    auto fmt = std::setw(6);
                    ss << seed << ',' << n << ',' << Gamma << ',' << theta << "," << (algo == InferenceAlgo::Comp ? "comp" : "dd") << ','
                       << m_min << ',' << m_max << ',' << m_lower << "," << m_upper << '\n';

                    found_valid += (m_lower != -1) || (m_upper != -1);
                }

                #pragma omp critical
                {
                    out << ss.str();
                }
            };

            if (!found_valid) {
                std::cout << "      -> stop here for this Gamma\n";
                break;
            }
        }
    }
}


void conv_const_test(std::ostream& out, double theta, unsigned Gamma) {
    auto steps = 50;
    auto seed = -1;

    const auto delta_dd = [&] {
        auto tt = theta / (1.0 - theta);
        if (ceil(tt) - tt < 1e-6)
            return std::max<size_t>(tt + 1, 2);
        return std::max<size_t>(std::ceil(tt), 2);
    }();

    const auto delta_comp = [&] {
        auto tt = 1.0 / (1.0 - theta);
        if (ceil(tt) - tt < 1e-6)
            return static_cast<size_t>(ceil(tt)) + 1;

        return static_cast<size_t>(ceil(tt));
    }();


    std::cout << "theta=" << theta << " Gamma=" << Gamma << "\n";
    for (size_t n = 1024; n < (1llu << 25); n *= 2) {
        std::cout << "  n=" << n << " log2(n)=" << std::log2(n) << std::endl;

#pragma omp parallel
        {
            auto &gen = get_threadlocal_prng();
            TestDesign td;
            std::stringstream ss;

            unsigned nthreads = omp_get_num_threads();
#pragma omp for nowait
            for (int r = 0; r < static_cast<int>(4.8 * nthreads); r++) {
                td.generate_design_gamma(n, tlx::div_ceil(delta_comp * n, Gamma), Gamma, gen);
                td.cough(theta, gen);

                if (delta_comp != delta_dd) {
                    report(ss, td, seed, td.algo_comp(), InferenceAlgo::Comp);

                    td.generate_design_gamma(n, tlx::div_ceil(delta_dd * n, Gamma), Gamma, gen);
                    td.cough(theta, gen);

                    report(ss, td, seed, td.algo_dd(), InferenceAlgo::DD);
                } else {
                    simulate_and_report(ss, td, seed);
                }
            }

#pragma omp critical
            {
                out << ss.str();
                out.flush();
            }
        }
    }
}

void scan_m_const_test(std::ostream& out, size_t n, double theta, unsigned Gamma) {
    auto steps = 50;

    std::vector<size_t> ms;
    {
        ms.reserve(steps + 1);
        const auto max_m = n;
        const auto min_m = n / Gamma / 4;

        for (unsigned i = 0; i <= steps; ++i) {
            ms.push_back(std::round(min_m * std::pow(2.0, std::log2(1.0 * max_m / min_m) / steps * i)));
        }
        ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
    }

    for (size_t m : ms) {
        std::cout << "n=" << n << " theta=" << theta << " m=" << m << std::endl;

#pragma omp parallel
        {
            auto &gen = get_threadlocal_prng();
            TestDesign td;
            std::stringstream ss;
	    
	    unsigned nthreads = omp_get_num_threads();
#pragma omp for nowait
            for (int r = 0; r < static_cast<int>(4.8 * nthreads); r++) {
                td.generate_design_gamma(n, m, Gamma, gen);
                td.cough(theta, gen);
                simulate_and_report(ss, td, 1);
                #pragma omp critical
                { std::cout << "."; }
            }

#pragma omp critical
            {
                out << ss.str();
                out.flush();
            }
        }
    }
}


void scan_m_const_test_sparse(std::ostream& out, size_t n, double theta, unsigned Gamma) {
    auto steps = 10;

    std::vector<size_t> ms;
    {
        ms.reserve(steps + 1);
        const auto max_m = tlx::div_ceil(2*n, Gamma);
        const auto min_m = tlx::div_ceil(2*n, Gamma + 1);

        for (unsigned i = 0; i <= steps; ++i) {
            ms.push_back(std::round(min_m * std::pow(2.0, std::log2(1.0 * max_m / min_m) / steps * i)));
        }
        ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
    }

    for (size_t m : ms) {
        std::cout << "n=" << n << " theta=" << theta << " m=" << m << " Delta=" << (1.0 * m * Gamma / n) << std::endl;

#pragma omp parallel
        {
            auto &gen = get_threadlocal_prng();
            TestDesign td;
            std::stringstream ss;

            unsigned nthreads = omp_get_num_threads();
#pragma omp for nowait
            for (int r = 0; r < static_cast<int>(4.8 * nthreads); r++) {
                td.generate_design_gamma(n, m, Gamma, gen, true);
                td.cough(theta, gen);
                const auto res = td.algo_dd();
                report(out, td, 1, res, InferenceAlgo::DD);
            }

#pragma omp critical
            {
                out << ss.str();
                out.flush();
            }
        }
    }
}


void scan_n_const_res(std::ostream& out, double theta, unsigned Delta) {
    for (size_t n = (1 << 10); n <= (1 << 24); n *= 2) {
        std::cout << "n=" << n << '\n';

#pragma omp parallel
        {
            auto &gen = get_threadlocal_prng();
            TestDesign td;
            std::stringstream ss;

            const auto nthreads = omp_get_num_threads();
#pragma omp for
            for (int repeat = 0; repeat < static_cast<int>(4.8 * nthreads); ++repeat) {
                auto run = [&] (const auto m, std::string series, InferenceAlgo algo) {
                    td.generate_design_delta(n, m, Delta, gen);
                    td.cough(theta, gen);

                    const auto res = td.algo(algo);
                    report(ss, td, -1, res, algo, series);
                };

                auto run_series = [&] (const auto m_base, InferenceAlgo algo) {
                    run(m_base / 2.0, "0.5", algo);
                    run(m_base * 1.0, "1.0", algo);
                    run(m_base * 1.5, "1.5", algo);
                    run(m_base * log(log(n)), "loglog", algo);
                    run(m_base * log(n), "log", algo);
                    run(m_base * log(n) * log(n), "log^2", algo);
                };

                const auto minf = static_cast<size_t>(std::ceil(std::max(
                    Delta * std::pow(n, theta * (1. + (1. - theta) / Delta)),
                    Delta * std::pow(n, theta * (1. +  1. / Delta))
                )));

                const auto mcomp = static_cast<size_t>(std::ceil(
                    Delta * std::pow(n, theta + 1. / Delta)
                ));

                if (minf >= 2.0)  run_series(minf, InferenceAlgo::DD);
                if (mcomp >= 2.0) run_series(mcomp, InferenceAlgo::Comp);
            }

#pragma omp critical
            {
                out << ss.str();
                out.flush();
            }
        };
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) { return -1; }

    const std::string task(argv[1]);

    if (task == "run1e6") {
        std::ofstream out(makeRandomFile("run1e6_"));
        print_sim_header(out);
        for (unsigned i = 0; i < 1000; i++) {
            for (double theta : {0.2, 0.4, 0.6}) {
                scan_m_const_test(out, 1000'000, theta, 3);
                scan_m_const_test(out, 1000'000, theta, 5);
            }
        }

        return 0;
    }

    if (task == "run1e7") {
        std::ofstream out(makeRandomFile("run1e7_"));
        print_sim_header(out);
        for (unsigned i = 0; i < 100; i++) {
            scan_m_const_test(out, 10'000'000, 0.2, 3);
            scan_m_const_test(out, 10'000'000, 0.6, 5);
        }

        return 0;
    }

    if (task == "run1e7sparse") {
        std::ofstream out(makeRandomFile("run1e7_sparse_"));
        print_sim_header(out);
        for (unsigned i = 0; i < 100; i++) {
            scan_m_const_test_sparse(out, 100'000, 0.2, 3);
        }

        return 0;
    }

    if (task == "search") {
        std::ofstream out(makeRandomFile("search_"));
        print_search_header(out);
        for (unsigned i = 0; i < 100; i++) {
            for (size_t n : {1'000, 10'000, 100'000}) {
                search_const_test(out, InferenceAlgo::Comp, n);
                search_const_test(out, InferenceAlgo::DD, n);
            }
        }

        return 0;
    }

    if (task == "scanres") {
        std::ofstream out(makeRandomFile("scanres_"));
        print_sim_header(out);

        for (unsigned i = 0; i < 100; i++) {
            scan_n_const_res(out, 0.3, 8);
            scan_n_const_res(out, 0.1, 3);
        }

        return 0;
    }

    if (task == "conv") {
        std::ofstream out(makeRandomFile("conv_"));
        print_sim_header(out);

        for (unsigned i = 0; i < 100; i++) {
            for(double theta : {0.6, 0.2, 0.5}) {
                for(unsigned Gamma : {5, 3}) {
                    conv_const_test(out, theta, Gamma);
                }
            }
        }

        return 0;
    }

    return -1;
}
