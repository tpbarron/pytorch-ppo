import torch


class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, action_shape):
        self.states = torch.zeros(num_steps + 1, 1, *obs_shape)
        self.rewards = torch.zeros(num_steps, 1, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1, 1)

        self.old_action_means = torch.zeros(num_steps, 1, *action_shape)
        self.old_action_log_stds = torch.zeros(num_steps, 1, *action_shape)
        self.old_action_stds = torch.zeros(num_steps, 1, *action_shape)

        # self.old_log_probs = torch.zeros(num_steps, 1, action_shape)
        self.returns = torch.zeros(num_steps + 1, 1, 1)
        self.actions = torch.LongTensor(num_steps, 1, *action_shape)
        self.masks = torch.zeros(num_steps, 1, 1)

    def cuda(self):
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.old_action_means = self.old_action_means.cuda()
        self.old_action_log_stds = self.old_action_log_stds.cuda()
        self.old_action_stds = self.old_action_stds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_state, action, value_pred, old_action_mean,
                old_action_log_std, old_action_std, reward, mask):
        self.states[step + 1].copy_(current_state)
        self.actions[step].copy_(action)
        self.value_preds[step].copy_(value_pred)
        self.old_action_means[step].copy_(old_action_mean)
        self.old_action_log_stds[step].copy_(old_action_log_std)
        self.old_action_stds[step].copy_(old_action_std)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step +
                                                                      1] * self.masks[step] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step] + self.rewards[step]
