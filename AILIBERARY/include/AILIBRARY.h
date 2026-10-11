//
// Created by User on 07.10.2026.
//

#ifndef TASKS_AILIBRARY_H
#define TASKS_AILIBRARY_H
#ifdef BUILDING_AI_DLL
    #define AI_API __declspec(dllexport)
#else
    #define AI_API __declspec(dllimport)
#endif
#include <vector>
using ll = long long;

struct AI_API Dense {
    ll n;
    std::vector<float> weights;
    std::vector<float> biases;

    Dense(ll size) : n(size), weights(size, 0.0f), biases(size) {}
};
struct AI_API InputDense : Dense {
    InputDense(ll size) : Dense(size) {}
    void set_input_values(const std::vector<float>& values) {
        if (values.size() == n) {
            weights = values;
        }
    }
};
struct AI_API OutputDense : Dense {
    OutputDense(ll size) : Dense(size){}

};

class  AI_API AI {
public:
    Dense FILL_BIASES(Dense& dense,int size);
    const Dense* get_hidden_dense(const Dense& dense);
    float get_weight(int idx, const Dense& dense);
    std::vector<float> get_vector(const Dense& current_dense);
    Dense FILL_WEIGHT(Dense& dense,int size);
    std::vector<float> mulDtoD(const std::vector<float>& last_dense_weights, const std::vector<float>& current_dense_weights);
    std::vector<float> forward_layer(const std::vector<float>& inputs, const std::vector<float>& weights, int next_layer_size);
    void apply_relu(std::vector<float>& vec);
    void apply_softmax(std::vector<float>& vec);
};

#endif // TASKS_AILIBRARY_H
