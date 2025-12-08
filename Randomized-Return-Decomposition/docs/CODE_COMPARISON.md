# TensorFlow vs PyTorch Code Comparison

This document shows side-by-side comparisons of key code sections to highlight the differences between the TensorFlow and PyTorch implementations.

## 1. Network Definition

### TensorFlow (rrd.py)
```python
def mlp_rrd(rrd_obs_ph, rrd_acts_ph, rrd_obs_next_ph):
    rrd_state_ph = tf.concat([rrd_obs_ph, rrd_acts_ph, rrd_obs_ph-rrd_obs_next_ph], axis=-1)
    with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
        r_dense1 = tf.layers.dense(rrd_state_ph, 256, activation=tf.nn.relu, name='r_dense1')
        r_dense2 = tf.layers.dense(r_dense1, 256, activation=tf.nn.relu, name='r_dense2')
        r = tf.layers.dense(r_dense2, 1, name='r')
    return r
```

### PyTorch (rrd_torch.py)
```python
class MLPRewardNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        state_dim = obs_dim * 2 + act_dim
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs, acts, obs_next):
        state = torch.cat([obs, acts, obs - obs_next], dim=-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

**Key Differences:**
- PyTorch uses `nn.Module` classes vs TensorFlow's functional API
- Explicit initialization in `__init__` vs variable scope
- `forward()` method vs direct function call
- `F.relu()` vs `tf.nn.relu`

---

## 2. Creating Placeholders/Inputs

### TensorFlow (rrd.py)
```python
def create_inputs(self):
    super().create_inputs()
    
    self.rrd_raw_obs_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
    self.rrd_raw_obs_next_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
    self.rrd_acts_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.acts_dims)
    self.rrd_rews_ph = tf.placeholder(tf.float32, [None, 1])
```

### PyTorch (rrd_torch.py)
```python
def preprocess_batch(self, batch):
    """Convert batch to tensors"""
    rrd_obs = torch.from_numpy(batch['rrd_obs']).float().to(self.device)
    rrd_obs_next = torch.from_numpy(batch['rrd_obs_next']).float().to(self.device)
    rrd_acts = torch.from_numpy(batch['rrd_acts']).float().to(self.device)
    rrd_rews = torch.from_numpy(batch['rrd_rews']).float().to(self.device)
    # ... normalize and return
```

**Key Differences:**
- No placeholders needed in PyTorch
- Direct tensor conversion from NumPy
- Explicit device placement (`.to(self.device)`)
- Done at runtime, not graph construction time

---

## 3. Loss Computation

### TensorFlow (rrd.py)
```python
def create_operators(self):
    super().create_operators()
    
    self.r_loss = tf.reduce_mean(tf.square(self.rrd-self.rrd_rews_ph))
    if self.args.rrd_bias_correction:
        n = self.args.rrd_sample_size
        self.r_var_single = tf.reduce_sum(
            tf.square(self.rrd_rews_pred-tf.reduce_mean(self.rrd_rews_pred, axis=1, keepdims=True)), 
            axis=1
        ) / (n-1)
        self.r_var = tf.reduce_mean(self.r_var_single*self.rrd_var_coef_ph/n)
        self.r_total_loss = self.r_loss - self.r_var
```

### PyTorch (rrd_torch.py)
```python
def train_r(self, batch):
    batch_tensors = self.preprocess_batch(batch)
    
    rrd_rews_pred = self.reward_net(...)
    rrd = torch.mean(rrd_rews_pred, dim=1)
    
    r_loss = F.mse_loss(rrd, batch_tensors['rrd_rews'])
    
    if self.args.rrd_bias_correction:
        n = self.args.rrd_sample_size
        r_var_single = torch.sum(
            torch.square(rrd_rews_pred - torch.mean(rrd_rews_pred, dim=1, keepdim=True)),
            dim=1
        ) / (n - 1)
        r_var = torch.mean(r_var_single * batch_tensors['rrd_var_coef'] / n)
        r_total_loss = r_loss - r_var
```

**Key Differences:**
- TensorFlow: Define loss in graph construction
- PyTorch: Compute loss in training function
- PyTorch uses `F.mse_loss()` vs `tf.reduce_mean(tf.square())`
- PyTorch: More readable with direct computation

---

## 4. Optimizer and Training

### TensorFlow (rrd.py)
```python
def create_operators(self):
    # ... loss computation ...
    
    self.r_optimizer = tf.train.AdamOptimizer(self.args.r_lr)
    self.r_train_op = self.r_optimizer.minimize(self.r_total_loss, var_list=get_vars('rrd/'))
    self.q_train_op = tf.group([self.q_train_op, self.r_train_op])

# Later, in training:
def train(self, batch):
    feed_dict = self.feed_dict(batch)
    info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
    return info
```

### PyTorch (rrd_torch.py)
```python
def create_optimizer(self):
    super().create_optimizer()
    self.r_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=self.args.r_lr)

def train_r(self, batch):
    batch_tensors = self.preprocess_batch(batch)
    self.reward_net.train()
    
    # Forward pass
    rrd_rews_pred = self.reward_net(...)
    r_total_loss = ...  # compute loss
    
    # Backward pass
    self.r_optimizer.zero_grad()
    r_total_loss.backward()
    self.r_optimizer.step()
    
    return info
```

**Key Differences:**
- TensorFlow: `sess.run()` with `feed_dict`
- PyTorch: Direct function calls with explicit backward pass
- PyTorch: `zero_grad()`, `backward()`, `step()` pattern
- PyTorch: More explicit and Pythonic

---

## 5. Target Network Updates

### TensorFlow (sac.py)
```python
def create_operators(self):
    self.target_update_op = tf.group([
        q_t.assign(self.args.polyak*q_t + (1.0-self.args.polyak)*q)
        for q, q_t in zip(get_vars('main/q_value'), get_vars('target/q_value'))
    ])

# Later:
def target_update(self):
    self.sess.run(self.target_update_op)
```

### PyTorch (sac_torch.py)
```python
def _target_update(self):
    """Soft update of target networks"""
    with torch.no_grad():
        for param, target_param in zip(self.q_value_1.parameters(), 
                                       self.q_value_1_target.parameters()):
            target_param.data.copy_(
                self.args.polyak * target_param.data + 
                (1.0 - self.args.polyak) * param.data
            )
```

**Key Differences:**
- TensorFlow: Pre-defined operation in graph
- PyTorch: Direct parameter manipulation
- PyTorch: Uses `torch.no_grad()` context
- PyTorch: More explicit with `.data.copy_()`

---

## 6. Normalizer

### TensorFlow (tf_utils.py)
```python
class Normalizer:
    def __init__(self, shape, sess, eps_std=1e-2, norm_clip=5.0):
        self.sess = sess
        # ... TensorFlow variables ...
        self.sum = tf.get_variable(name='sum', shape=self.shape, trainable=False)
        self.add_sum = tf.placeholder(tf.float32, self.shape)
        self.update_array_op = tf.group(self.sum.assign_add(self.add_sum), ...)
        
    def normalize(self, inputs_ph):
        return tf.clip_by_value((inputs_ph-self.mean)/self.std, -self.norm_clip, self.norm_clip)
    
    def update(self, inputs):
        feed_dict = {self.add_sum: np.sum(inputs, axis=0), ...}
        self.sess.run(self.update_array_op, feed_dict)
```

### PyTorch (torch_utils.py)
```python
class Normalizer:
    def __init__(self, shape, device='cpu', eps_std=1e-2, norm_clip=5.0):
        self.device = device
        # ... PyTorch tensors ...
        self.sum = torch.zeros(shape, dtype=torch.float32, device=device)
        
    def normalize(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        return torch.clamp((inputs - self.mean) / self.std, -self.norm_clip, self.norm_clip)
    
    def update(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        self.sum += torch.from_numpy(np.sum(inputs, axis=0)).float().to(self.device)
```

**Key Differences:**
- TensorFlow: Requires session and placeholders
- PyTorch: Direct tensor operations
- PyTorch: Explicit device management
- PyTorch: Simpler update logic

---

## 7. Model Saving/Loading

### TensorFlow (base.py)
```python
def save_model(self, save_path):
    with self.graph.as_default():
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

def load_model(self, load_path):
    with self.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
```

### PyTorch (base_torch.py)
```python
def save_model(self, save_path):
    state_dict = {'networks': {}}
    for name, module in self.__dict__.items():
        if isinstance(module, nn.Module):
            state_dict['networks'][name] = module.state_dict()
    if self.obs_normalizer is not None:
        state_dict['normalizer'] = self.obs_normalizer.state_dict()
    torch.save(state_dict, save_path)

def load_model(self, load_path):
    state_dict = torch.load(load_path, map_location=self.device)
    for name, params in state_dict['networks'].items():
        if hasattr(self, name):
            getattr(self, name).load_state_dict(params)
    if 'normalizer' in state_dict and self.obs_normalizer is not None:
        self.obs_normalizer.load_state_dict(state_dict['normalizer'])
```

**Key Differences:**
- TensorFlow: Uses `tf.train.Saver()`
- PyTorch: Uses `torch.save()` with state dicts
- PyTorch: More flexible with `state_dict()`
- PyTorch: Explicit device mapping with `map_location`

---

## 8. Action Sampling (SAC)

### TensorFlow (sac.py)
```python
with tf.variable_scope('policy'):
    self.pi = mlp_policy(self.obs_ph)
    self.pi_sample = self.pi.mean+self.pi_noise_ph*self.pi.std
    self.pi_act = tf.tanh(self.pi_sample)
    self.pi_log_p = self.pi.log_p(self.pi_sample) - \
        tf.reduce_sum(tf.log(1-tf.square(self.pi_act)+1e-6), axis=-1, keepdims=True)

def step(self, obs, explore=False):
    noise = np.random.normal(0.0, 1.0, size=self.args.acts_dims) if explore else np.zeros(...)
    feed_dict = {self.raw_obs_ph: [obs], self.pi_noise_ph: [noise]}
    action, info = self.sess.run([self.pi_act, self.step_info], feed_dict)
    return action[0]
```

### PyTorch (sac_torch.py)
```python
def get_action_and_log_prob(self, pi_dist, noise):
    pi_sample = pi_dist.mean + noise * pi_dist.std
    pi_act = torch.tanh(pi_sample)
    pi_log_p = pi_dist.log_p(pi_sample) - torch.sum(
        torch.log(1 - torch.square(pi_act) + 1e-6), dim=-1, keepdim=True
    )
    return pi_act, pi_log_p

def step(self, obs, explore=False, test_info=False):
    self.policy.eval()
    with torch.no_grad():
        obs_tensor = self.normalize_obs(np.array([obs]))
        pi_dist = self.policy(obs_tensor)
        noise = torch.randn_like(pi_dist.mean) if explore else torch.zeros_like(pi_dist.mean)
        pi_act, _ = self.get_action_and_log_prob(pi_dist, noise)
        action = pi_act.cpu().numpy()[0]
    return action
```

**Key Differences:**
- TensorFlow: Noise passed via placeholder
- PyTorch: Noise generated in function
- PyTorch: Uses `torch.no_grad()` for inference
- PyTorch: Explicit eval mode with `.eval()`

---

## Summary Table

| Aspect | TensorFlow 1.x | PyTorch |
|--------|---------------|---------|
| **Execution** | Graph-based | Eager |
| **Networks** | Functions with variable scopes | `nn.Module` classes |
| **Inputs** | Placeholders | Direct tensors |
| **Training** | `sess.run()` + `feed_dict` | `.backward()` + `.step()` |
| **Verbosity** | More boilerplate | More concise |
| **Debugging** | Harder (lazy execution) | Easier (eager execution) |
| **Pythonic** | Less | More |
| **Performance** | Good | Similar or better |

---

## Migration Pattern

When converting TensorFlow code to PyTorch:

1. **Network Definition**: Convert to `nn.Module` classes
2. **Placeholders**: Remove, use function arguments
3. **Operations**: Define in functions, not graph
4. **Session**: Remove all session-related code
5. **Training**: Use standard PyTorch training loop
6. **Variables**: Use parameters and buffers
7. **Device**: Explicit `.to(device)` calls

This conversion maintains the same algorithm logic while using PyTorch's more modern and Pythonic API.






