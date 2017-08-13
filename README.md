# PyTorch implementation of PPO

This is a PyTorch implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

This is code mostly ported from the [OpenAI baselines implementation](https://github.com/openai/baselines) but currently does not optimize each batch for several epochs. I will add this soon.

## Usage

```
python main.py --env-name Walker2d-v1
```
## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Todo

- [ ] Add multiple epochs per batch
- [ ] Test results compared to baselines code
