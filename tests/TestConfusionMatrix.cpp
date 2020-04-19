#include <tlx/die.hpp>
#include <ConfusionMatrix.hpp>

int main() {
    std::unordered_set<int> truth = {0,1,2,3};
    std::unordered_set<int> half  = {2,3};
    std::unordered_set<int> overlap = {2,3,4,5};

    {
        auto res = ConfusionMatrix::compute(truth, half, 8);
        die_unequal(res.true_positive(),  2u);
        die_unequal(res.true_negative(),  4u);
        die_unequal(res.false_positive(), 0u);
        die_unequal(res.false_negative(), 2u);
        die_unequal_eps(res.accuracy(), 6. / 8., 0.001);
    }

    {
        auto res = ConfusionMatrix::compute(truth, overlap, 8);
        die_unequal(res.true_positive(),  2u);
        die_unequal(res.true_negative(),  2u);
        die_unequal(res.false_positive(), 2u);
        die_unequal(res.false_negative(), 2u);
        die_unequal_eps(res.accuracy(), 0.5, 0.001);
    }


    return 0;
}