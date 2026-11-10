// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Config struct for SGD+momentum hyperparameters.
#derive(Debug)
pub(all) struct SgdMomentumConfig {
  lr : Float
  momentum : Float
  weight_decay : Float
  max_grad_norm : Float
}

///|
/// Config struct for Adam hyperparameters.
#derive(Debug)
pub(all) struct AdamConfig {
  lr : Float
  beta1 : Float
  beta2 : Float
  one_minus_beta1 : Float
  one_minus_beta2 : Float
  bias_correction1 : Float
  bias_correction2 : Float
  eps : Float
  weight_decay : Float
  max_grad_norm : Float
}

///|
/// Config struct for RMSprop hyperparameters.
#derive(Debug)
pub(all) struct RmspropConfig {
  lr : Float
  alpha : Float
  one_minus_alpha : Float
  eps : Float
  weight_decay : Float
  momentum : Float
  has_momentum : Bool
  max_grad_norm : Float
}

///|
/// Backend trait for fused batch SGD+momentum with auto gradient clipping.
/// Fine-grained: only the method momentum_sgd actually uses.
/// Add new optimizers by defining new traits — no monolithic OptimStep.
pub(open) trait BatchSgdMomentumAutoclip {
  fn batch_sgd_momentum_autoclip(
    params : Array[Self],
    grads : Array[Self],
    velocities : Array[Self],
    config : SgdMomentumConfig,
  ) -> Unit
}

///|
/// Backend trait for fused batch Adam with auto gradient clipping.
pub(open) trait BatchAdamAutoclip {
  fn batch_adam_autoclip(
    params : Array[Self],
    grads : Array[Self],
    m : Array[Self],
    v : Array[Self],
    config : AdamConfig,
  ) -> Unit
}

///|
/// Backend trait for fused batch RMSprop with auto gradient clipping.
pub(open) trait BatchRmspropAutoclip {
  fn batch_rmsprop_autoclip(
    params : Array[Self],
    grads : Array[Self],
    v : Array[Self],
    buf : Array[Self],
    config : RmspropConfig,
  ) -> Unit
}

///|
/// Momentum SGD optimizer with gradient clipping and weight decay.
/// Generic over any Tensor type that implements BatchSgdMomentumAutoclip.
///
/// Returns an Optimizer[T] whose closure captures velocities as internal state.
pub fn[T : @tensor.Tensor + BatchSgdMomentumAutoclip] momentum_sgd(
  params : Array[@grad.Grad[T]],
  momentum : Double,
  weight_decay : Double,
  max_grad_norm : Double,
) -> Optimizer[T] {
  let velocities = params.map(fn(p : @grad.Grad[T]) -> T {
    @tensor.Tensor::zeros_like(p.value)
  })
  let momentum_f = Float::from_double(momentum)
  let weight_decay_f = Float::from_double(weight_decay)
  let max_grad_norm_f = Float::from_double(max_grad_norm)
  {
    params,
    _step: fn(params : Array[@grad.Grad[T]], lr : Float) -> Unit {
      let mut has_grad = false
      let n = params.length()
      let param_vals : Array[T] = Array::makei(n, fn(i) { params[i].value })
      let grad_vals : Array[T] = Array::makei(n, fn(i) {
        match params[i].grad {
          Some(Some(g)) => {
            has_grad = true
            g
          }
          Some(None) | None => velocities[i]
        }
      })

      if has_grad == false {
        return
      }

      BatchSgdMomentumAutoclip::batch_sgd_momentum_autoclip(
        param_vals,
        grad_vals,
        velocities,
        {
          lr,
          momentum: momentum_f,
          weight_decay: weight_decay_f,
          max_grad_norm: max_grad_norm_f,
        },
      )
    },
  }
}
