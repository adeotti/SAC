#include <iostream>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <cassert>
#include <ATen/ATen.h>


namespace py = pybind11;
namespace F = torch::nn::functional;
using torch::Tensor;


struct gaussian {
    Tensor mean, std;
    gaussian(Tensor loc, Tensor scale): loc(loc), scale(scale){}

    Tensor sample(){
        return 0;
    }
};


Tensor compute_q_target(Tensor q1, Tensor q2, Tensor log_nx_actions, Tensor reward, Tensor done, Tensor alpha, float gamma, float exp = 1.0){
    assert(
            (q1.sizes() == q2.sizes()) && (q2.sizes() == log_nx_actions.sizes()) &&
            (log_nx_actions.sizes() == reward.sizes()) && (reward.sizes() == done.sizes())
    );
    auto min_ = torch::min(q1, q2);
    auto discount = std::pow(gamma, exp);
    return reward + discount * (1-done) * (min_ - alpha.detach() * log_nx_actions); // [1024, 1]
}

Tensor get_critics_loss(Tensor q1_pred, Tensor q2_pred, Tensor q_target){
    assert((q1_pred.sizes() == q2_pred.sizes()) && (q2_pred.sizes() == q_target.sizes()));
    return F::smooth_l1_loss(q1_pred, q_target) + F::smooth_l1_loss(q2_pred, q_target);
}

std::pair<Tensor, Tensor> get_policy_loss(Tensor q1_pred, Tensor q2_pred, Tensor alpha, Tensor log_pi){
    assert((q1_pred.sizes() == q2_pred.sizes()) && (q2_pred.sizes() == log_pi.sizes()));
    Tensor min_ = torch::min(q1_pred, q2_pred);
    return {((alpha.detach() * log_pi) - min_).mean(), min_.squeeze()};
}

Tensor tune_alpha(Tensor log_alpha, Tensor log_pi, Tensor entropy_target, torch::optim::Optimizer& optim){
    Tensor alpha_loss = (log_alpha * (-log_pi - entropy_target).detach()).mean();
    optim.zero_grad();
    alpha_loss.backward();
    optim.step();
    return log_alpha.exp();
}

Tensor relabel_goals(Tensor hl_goals, Tensor states, Tensor nx_states, Tensor actions, int hl_action_dim){
    torch::NoGradGuard no_grad;
    
    Tensor states = states.unsqueeze(2);
    Tensor sample_1 = hl_goals.unsqueeze(1).unsqueeze(1);
    Tensor sample_2 = extract_goals(_nx_states.unsqueeze(1)); // implement extract_goals method
    Tensor sample_3 = Normal(loc=sample_2, scale=0.5).sample((8,)).permute(1, 2, 0, -1); // implement Normal distribution

    Tensor g_stack = torch::cat({sample_1, sample_2, sample_3}, /*dim*/2);
    Tensor g_stack = g_stack.expand(-1, 10, -1, -1);               // [1024,  1, 10,  6] > [1024, 10, 10,  6]
    Tensor actions = actions.unsqueeze(2).expand(-1, -1, 10, -1);  // [1024, 10,  1,  9] > [1024, 10, 10,  9]
    Tensor states = states.expand(-1, -1, 10, -1);                 // [1024, 10,  1, 81] > [1024, 10, 10, 81]
    
    Tensor log = llp.evaluate_actions();                                             // [1024, 10, 10, 1]
    Tensor log = torch::sum(log, /*dim*/{1}, /*keepdim*/true);  // [1024,  1, 10, 1]
    Tensor argmax = torch::argmax(log, {2}, true);              // [1024,  1,  1, 1] 

    Tensor idx = argmax::expand(-1, 10, -1, hl_action_dim);              // [1024, 10,  1, 6] 
    Tensor hl_goals = torch::take_along_dim(g_stack, idx, 2).squeeze(2); // [1024, 10,  6]
    return _hl_goals;
}


struct LowLevelTrainer{
    Tensor states, nx_states, local_reward, dones, actions, hl_goals, obs_goals, ll_alpha;
    const float tau;
    torch::nn::Module llp, llp_cpu, q1_net, q2_net, q1_target_net, q2_target_net;
    torch::optim::Optimizer llp_optim, ql_optim;

    LowLevelTrainer(Tensor states, Tensor nx_states, Tensor local_reward, Tensor dones, Tensor actions, Tensor hl_goals, Tensor obs_goals)
        : states(states), nx_states(nx_states), local_reward(local_reward), dones(dones), actions(actions), hl_goals(hl_goals), obs_goals(obs_goals){}
    
    Tensor q_target;
    {
        torch::NoGradGuard no_grad;

        Tensor [nx_actions, Tensor log_nx_actions] = llp(nx_states, hl_goals);
        Tensor q1_pred = q1_target_net(nx_states, nx_actions, hl_goals);
        Tensor q2_pred = q2_target_net(nx_states, nx_actions, hl_goals);
        Tensor q_target = compute_q_target(q1_pred, q2_pred, log_nx_actions, _local_reward, _dones, ll_alpha)

    };
    
    Tensor q1_pred = q1_target_net(states, actions, hl_goals);
    Tensor q2_pred = q2_target_net(states, actions, hl_goals);
    Tensor ll_q_loss = get_critics_loss(q1_pred, q2_pred, q_target);
    ql_optim.zero_grad();
    ll_q_loss.backward();
    ql_optim.step();

    Tensor [new_action, log_pi, n] = llp(states, hl_goals);
    Tensor q1_pred = q1_net(states, new_action, hl_goals);
    Tensor q2_pred = q2_net(states, new_action, hl_goals);
    Tensor [ll_policy_loss, min_q] = get_policy_loss(q1_pred, q2_pred, ll_alpha, log_pi);
    llp_optim.zero_grad();
    ll_policy_loss.backward();
    llp_optim.step();

    {
        torch::NoGradGuard no_grad;
        
        auto p1 = ql_target->parameters();
        auto p2 = ql_target->parameters();
        at::_foreach_lerp_(p1, p2, tau);

        auto gpu_params = llp->parameters();
        auto cpu_params = llp_cpu->parameters();
        for (auto [gpu, cpu] : std::views::zip(gpu_params, cpu_params)){cpu.copy_(gpu)}    
    };
    
    auto ll_alpha = tune_alpha(ll_log_alpha, log_pi, ll_entropy_target, ll_alpha_optim);

    return {0}
};


struct HighLevelTrainer{
    Tensor states, nx_states, reward, dones, actions, hl_goals, obs_goals, hl_alpha;

    const float tau;
    torch::nn::Module hlp, hlp_cpu, q1_net, q2_net, q1_target_net, q2_target_net;
    torch::optim::Optimizer hlp_optim, ql_optim;
    
    HighLevelTrainer(Tensor states, Tensor nx_states, Tensor reward, Tensor dones, Tensor actions, Tensor hl_goals, Tensor hl_alpha):
        states(state), nx_states(nx_states), reward(reward), dones(dones), actions(actions), hl_goals(hl_goals), hl_alpha(hl_alpha){}

    Tensor hl_goals = relabel_goals(hl_goals, states, nx_states, actions);
    Tensor states = states.select();
    hl_goals = hl_goals.select();

    Tensor q_target;
    {
        torch::NoGradGuard no_grad;

        Tensor [nx_actions, log_nx_actions, n] = hlp(nx_states);
        Tensor q1 = q1_target_net(nx_states, nx_actions);
        Tensor q2 = q2_target_net(nx_states, nx_actions);
        Tensor q_target = compute_q_target(q1, q2, log_nx_actions, reward, dones, hl_alpha, 10);
    }

    Tensor q1_pred = q1_target_net(states, hl_goals);
    Tensor q2_pred = q2_target_net(states, hl_goals);
    Tensor hl_q_loss = get_critics_loss(q1_pred, q2_pred, q_target);
    ql_optim.zero_grad();
    hl_q_loss.backward();
    ql_optim.step();

    Tensor [new_action, log_pi, n] = llp(states, hl_goals);
    Tensor q1_pred = q1_net(states, new_action, hl_goals);
    Tensor q2_pred = q2_net(states, new_action, hl_goals);
    Tensor [hl_policy_loss, min_q] = get_policy_loss(q1_pred, q2_pred, ll_alpha, log_pi);
    hlp_optim.zero_grad();
    hl_policy_loss.backward();
    hlp_optim.step();

    {
        torch::NoGradGuard no_grad;
        
        auto p1 = ql_target->parameters();
        auto p2 = ql_target->parameters();
        at::_foreach_lerp_(p1, p2, tau);

        auto gpu_params = llp->parameters();
        auto cpu_params = llp_cpu->parameters();
        for (auto [gpu, cpu] : std::views::zip(gpu_params, cpu_params)){cpu.copy_(gpu)}    
    };
    
    auto hl_alpha = tune_alpha(hl_log_alpha, log_pi, hl_entropy_target, hl_alpha_optim);
    
    return {0};
};

