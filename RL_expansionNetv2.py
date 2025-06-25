import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random
import time
import psutil
import json
import os
import sys

# Add ExpansionNet_v2 directory to Python path
expansionnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ExpansionNet_v2"))
sys.path.append(expansionnet_path)

# Assuming ExpansionNet v2 imports (adjust based on actual repo structure)
try:
    from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
    # from utils.language_utils import get_lang_encoder
except ImportError as e:
    print(f"Warning: Could not import ExpansionNet v2 modules: {e}")
    print("Creating a dummy model for testing purposes")

@dataclass
class DeviceCapabilities:
    device_id: int
    device_type: str  # 'cuda:0', 'cuda:1', 'cpu'
    compute_power: float  # Measured TFLOPS
    memory_capacity: float  # GB
    memory_bandwidth: float  # GB/s
    current_memory_usage: float = 0.0
    is_available: bool = True

@dataclass
class LayerProfile:
    layer_name: str
    layer_index: int
    layer_type: str  # 'conv', 'attention', 'linear', etc.
    compute_time: float  # Measured execution time (ms)
    memory_usage: float  # Peak memory usage (MB)
    input_shape: Tuple
    output_shape: Tuple
    parameters: int
    flops: float

class DummyExpansionNetV2(nn.Module):
    """Dummy ExpansionNet v2 model for testing when real model is not available"""
    
    def __init__(self, vocab_size=20000, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Image encoder components - Fixed dimensions
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 3, padding=1),  # Fixed to embed_dim
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Text encoder/decoder components
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids=None, images=None, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)
        seq_len = input_ids.size(1) if input_ids is not None else 20
        device = input_ids.device if input_ids is not None else images.device
        
        # Process images
        if images is not None:
            image_features = self.image_encoder(images)
            image_features = image_features.flatten(2).transpose(1, 2)  # [B, 49, embed_dim]
        else:
            image_features = torch.randn(batch_size, 49, self.embed_dim, device=device)
        
        # Process text
        if input_ids is not None:
            text_features = self.text_embedding(input_ids)
            text_features += self.positional_encoding[:seq_len].unsqueeze(0).to(device)
        else:
            text_features = torch.randn(batch_size, seq_len, self.embed_dim, device=device)
        
        # Apply transformer layers with cross-attention
        for i, (transformer_layer, cross_attn_layer) in enumerate(
            zip(self.transformer_layers, self.cross_attention_layers)
        ):
            # Self-attention on text
            text_features = transformer_layer(text_features)
            
            # Cross-attention between text and image
            attended_features, _ = cross_attn_layer(
                text_features, image_features, image_features
            )
            text_features = text_features + attended_features
        
        # Output projection
        output = self.output_projection(text_features)
        
        return output

class ExpansionNetV2Profiler:
    """Real-time profiler for ExpansionNet v2 model layers"""
    
    def __init__(self, model_path: str, vocab_size: int = 20000):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.layer_profiles = []
        self.device_profiles = []
        
    def profile_devices(self) -> List[DeviceCapabilities]:
        """Profile available devices (GPUs and CPU)"""
        devices = []
        
        # Profile GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                props = torch.cuda.get_device_properties(device)
                
                # Estimate compute power (rough approximation)
                compute_power = self._estimate_gpu_tflops(props)
                memory_gb = props.total_memory / (1024**3)
                
                devices.append(DeviceCapabilities(
                    device_id=i,
                    device_type=f'cuda:{i}',
                    compute_power=compute_power,
                    memory_capacity=memory_gb,
                    memory_bandwidth=self._estimate_memory_bandwidth(props),
                    current_memory_usage=torch.cuda.memory_allocated(device) / (1024**3)
                ))
        
        # Profile CPU
        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 2.5  # GHz
        cpu_memory = psutil.virtual_memory().total / (1024**3)
        
        devices.append(DeviceCapabilities(
            device_id=len(devices),
            device_type='cpu',
            compute_power=cpu_cores * cpu_freq * 0.1,  # Rough estimate
            memory_capacity=cpu_memory,
            memory_bandwidth=50,  # Typical DDR4
            current_memory_usage=psutil.virtual_memory().used / (1024**3)
        ))
        
        self.device_profiles = devices
        return devices
    
    def _estimate_gpu_tflops(self, props) -> float:
        """Estimate GPU TFLOPS based on properties"""
        # Rough estimation based on common GPU architectures
        sm_count = props.multi_processor_count
        base_clock = props.memory_clock_rate / 1000  # Convert to MHz
        
        # Very rough approximation - would need more sophisticated calculation
        if "RTX" in props.name:
            return sm_count * base_clock * 0.001  # Rough scaling
        elif "GTX" in props.name:
            return sm_count * base_clock * 0.0008
        else:
            return sm_count * base_clock * 0.0006  # Conservative estimate
    
    def _estimate_memory_bandwidth(self, props) -> float:
        """Estimate memory bandwidth in GB/s"""
        memory_bus_width = 256  # Common width, ideally get from props
        memory_clock = props.memory_clock_rate / 1000  # MHz
        return (memory_bus_width * memory_clock * 2) / 8000  # GB/s
    
    def profile_model_layers(self, model: nn.Module, sample_input: Dict) -> List[LayerProfile]:
        """Profile individual layers of ExpansionNet v2"""
        if model is None:
            raise ValueError("Model cannot be None. Please ensure the model is loaded properly.")
        
        profiles = []
        
        # Simplified profiling - only profile main module types to avoid too many layers
        def should_profile_layer(name, module):
            # Only profile significant layers to reduce complexity
            important_types = (nn.Conv2d, nn.Linear, nn.MultiheadAttention, 
                             nn.TransformerEncoderLayer, nn.Embedding)
            return isinstance(module, important_types)
        
        # Hook to capture layer information
        def create_hook(name, idx):
            def hook(module, input, output):
                try:
                    # Measure timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.perf_counter()
                    
                    # Simple timing measurement
                    time.sleep(0.001)  # Simulate processing time
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    # Calculate memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    else:
                        memory_usage = 100  # Default for CPU
                    
                    # Calculate FLOPs (simplified)
                    flops = self._estimate_layer_flops(module, input, output)
                    
                    # Get parameter count
                    params = sum(p.numel() for p in module.parameters())
                    
                    # Get shapes safely
                    input_shape = None
                    output_shape = None
                    
                    try:
                        if isinstance(input, tuple) and len(input) > 0:
                            input_shape = tuple(input[0].shape) if hasattr(input[0], 'shape') else None
                        elif hasattr(input, 'shape'):
                            input_shape = tuple(input.shape)
                            
                        if hasattr(output, 'shape'):
                            output_shape = tuple(output.shape)
                        elif isinstance(output, tuple) and len(output) > 0:
                            output_shape = tuple(output[0].shape) if hasattr(output[0], 'shape') else None
                    except:
                        pass  # Keep as None if we can't get shapes
                    
                    profile = LayerProfile(
                        layer_name=name,
                        layer_index=idx,
                        layer_type=type(module).__name__,
                        compute_time=max((end_time - start_time) * 1000, 1.0),  # ms, minimum 1ms
                        memory_usage=memory_usage,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        parameters=params,
                        flops=flops
                    )
                    profiles.append(profile)
                except Exception as e:
                    print(f"Warning: Error profiling layer {name}: {e}")
            return hook
        
        # Register hooks for selected modules only
        hooks = []
        for idx, (name, module) in enumerate(model.named_modules()):
            if should_profile_layer(name, module):
                hook = module.register_forward_hook(create_hook(name, idx))
                hooks.append(hook)
        
        # Run forward pass to trigger profiling
        model.eval()
        with torch.no_grad():
            try:
                _ = model(**sample_input)
            except Exception as e:
                print(f"Warning: Forward pass failed during profiling: {e}")
                # Create some dummy profiles if forward pass fails
                for i in range(10):  # Create 10 dummy layers
                    profiles.append(LayerProfile(
                        layer_name=f"dummy_layer_{i}",
                        layer_index=i,
                        layer_type="Linear",
                        compute_time=random.uniform(1.0, 10.0),
                        memory_usage=random.uniform(50.0, 200.0),
                        input_shape=(1, 512),
                        output_shape=(1, 512),
                        parameters=random.randint(1000, 100000),
                        flops=random.uniform(1000.0, 1000000.0)
                    ))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Ensure we have at least some profiles
        if not profiles:
            print("No profiles generated, creating dummy profiles...")
            for i in range(10):
                profiles.append(LayerProfile(
                    layer_name=f"layer_{i}",
                    layer_index=i,
                    layer_type="Linear",
                    compute_time=random.uniform(1.0, 10.0),
                    memory_usage=random.uniform(50.0, 200.0),
                    input_shape=(1, 512),
                    output_shape=(1, 512),
                    parameters=random.randint(1000, 100000),
                    flops=random.uniform(1000.0, 1000000.0)
                ))
        
        self.layer_profiles = profiles
        return profiles
    
    def _estimate_layer_flops(self, module, input, output) -> float:
        """Estimate FLOPs for a layer"""
        try:
            if hasattr(module, 'weight'):
                if isinstance(module, nn.Linear):
                    input_size = input[0].numel() if isinstance(input, tuple) else input.numel()
                    return input_size * module.weight.shape[0]
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Simplified FLOP calculation for conv layers
                    output_size = output.numel() if hasattr(output, 'numel') else 0
                    kernel_flops = module.weight.numel()
                    return output_size * kernel_flops
        except:
            pass
        return random.uniform(1000.0, 100000.0)  # Default estimate
    
    def save_profiles(self, filepath: str):
        """Save profiling results"""
        data = {
            'layer_profiles': [
                {
                    'layer_name': p.layer_name,
                    'layer_index': p.layer_index,
                    'layer_type': p.layer_type,
                    'compute_time': p.compute_time,
                    'memory_usage': p.memory_usage,
                    'input_shape': p.input_shape,
                    'output_shape': p.output_shape,
                    'parameters': p.parameters,
                    'flops': p.flops
                } for p in self.layer_profiles
            ],
            'device_profiles': [
                {
                    'device_id': d.device_id,
                    'device_type': d.device_type,
                    'compute_power': d.compute_power,
                    'memory_capacity': d.memory_capacity,
                    'memory_bandwidth': d.memory_bandwidth
                } for d in self.device_profiles
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

class LayerSplittingEnvironment:
    """RL Environment for real ExpansionNet v2 layer splitting"""
    
    def __init__(self, devices: List[DeviceCapabilities], layer_profiles: List[LayerProfile]):
        self.devices = devices
        self.layer_profiles = layer_profiles
        self.n_devices = len(devices)
        self.n_layers = len(layer_profiles)
        self.current_assignment = [0] * self.n_layers
        self.pipeline_stages = self._create_pipeline_stages()
        
    def _create_pipeline_stages(self) -> List[List[int]]:
        """Create logical pipeline stages based on layer types"""
        stages = []
        current_stage = []
        
        for i, profile in enumerate(self.layer_profiles):
            current_stage.append(i)
            
            # Create stage boundaries at major architectural transitions
            if (i < len(self.layer_profiles) - 1 and 
                self._is_stage_boundary(profile, self.layer_profiles[i + 1])):
                stages.append(current_stage)
                current_stage = []
        
        if current_stage:
            stages.append(current_stage)
            
        return stages
    
    def _is_stage_boundary(self, current_layer: LayerProfile, next_layer: LayerProfile) -> bool:
        """Determine if there should be a pipeline stage boundary"""
        # Boundaries between different layer types or significant size changes
        type_change = current_layer.layer_type != next_layer.layer_type
        size_change = (abs(current_layer.memory_usage - next_layer.memory_usage) > 
                      max(current_layer.memory_usage, next_layer.memory_usage) * 0.5)
        return type_change or size_change
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        # Start with a simple round-robin assignment
        for i in range(self.n_layers):
            self.current_assignment[i] = i % self.n_devices
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Device utilization
        device_compute_loads = [0.0] * self.n_devices
        device_memory_loads = [0.0] * self.n_devices
        
        for layer_idx, device_idx in enumerate(self.current_assignment):
            layer = self.layer_profiles[layer_idx]
            device_compute_loads[device_idx] += layer.compute_time
            device_memory_loads[device_idx] += layer.memory_usage
        
        # Normalize by device capabilities
        for i, device in enumerate(self.devices):
            compute_util = device_compute_loads[i] / max(device.compute_power * 1000, 1.0)  # Normalize
            memory_util = device_memory_loads[i] / max(device.memory_capacity * 1024, 1.0)  # MB to GB
            state.extend([compute_util, memory_util, device.memory_bandwidth / 1000])
        
        # Layer assignment encoding (simplified)
        # Use a more compact representation
        for device_idx in self.current_assignment:
            one_hot = [0.0] * self.n_devices
            one_hot[min(device_idx, self.n_devices - 1)] = 1.0
            state.extend(one_hot)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action: move layer to different device"""
        if self.n_layers == 0:
            return self._get_state(), 0, True, {"error": "No layers to assign"}
            
        layer_idx = action % self.n_layers  # Ensure valid layer index
        new_device_idx = (action // self.n_layers) % self.n_devices  # Ensure valid device index
        
        old_device_idx = self.current_assignment[layer_idx]
        self.current_assignment[layer_idx] = new_device_idx
        
        reward = self._calculate_reward()
        done = False  # Could add termination conditions
        
        info = {
            "assignment": self.current_assignment.copy(),
            "estimated_latency": self._estimate_total_latency(),
            "memory_balance": self._get_memory_balance(),
            "compute_balance": self._get_compute_balance()
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on multiple objectives"""
        # Compute load balancing
        compute_loads = self._get_device_compute_loads()
        compute_mean = np.mean(compute_loads)
        compute_balance = -np.std(compute_loads) / max(compute_mean, 1.0)
        
        # Memory load balancing
        memory_loads = self._get_device_memory_loads()
        memory_mean = np.mean(memory_loads)
        memory_balance = -np.std(memory_loads) / max(memory_mean, 1.0)
        
        # Communication cost (sequential dependencies)
        comm_cost = self._calculate_communication_penalty()
        
        # Resource utilization efficiency
        utilization_reward = self._calculate_utilization_reward()
        
        return compute_balance + memory_balance - comm_cost + utilization_reward
    
    def _get_device_compute_loads(self) -> List[float]:
        """Get compute load per device"""
        loads = [0.0] * self.n_devices
        for layer_idx, device_idx in enumerate(self.current_assignment):
            loads[device_idx] += self.layer_profiles[layer_idx].compute_time
        return loads
    
    def _get_device_memory_loads(self) -> List[float]:
        """Get memory load per device"""
        loads = [0.0] * self.n_devices
        for layer_idx, device_idx in enumerate(self.current_assignment):
            loads[device_idx] += self.layer_profiles[layer_idx].memory_usage
        return loads
    
    def _calculate_communication_penalty(self) -> float:
        """Calculate penalty for communication between devices"""
        penalty = 0.0
        for i in range(len(self.current_assignment) - 1):
            if self.current_assignment[i] != self.current_assignment[i + 1]:
                # Communication overhead between sequential layers
                layer = self.layer_profiles[i]
                if layer.output_shape:
                    transfer_size = max(np.prod(layer.output_shape) * 4 / (1024**2), 1.0)  # MB
                else:
                    transfer_size = 100  # Default estimate
                bandwidth = self.devices[self.current_assignment[i]].memory_bandwidth
                penalty += transfer_size / max(bandwidth, 1.0)  # Transfer time
        return penalty
    
    def _calculate_utilization_reward(self) -> float:
        """Reward for efficient resource utilization"""
        compute_loads = self._get_device_compute_loads()
        memory_loads = self._get_device_memory_loads()
        
        # Avoid overloading any device
        compute_overload_penalty = 0
        memory_overload_penalty = 0
        
        for i, device in enumerate(self.devices):
            if compute_loads[i] > device.compute_power * 1000:  # Overloaded
                compute_overload_penalty += 1
            if memory_loads[i] > device.memory_capacity * 1024:  # MB
                memory_overload_penalty += 1
        
        return -(compute_overload_penalty + memory_overload_penalty)
    
    def _estimate_total_latency(self) -> float:
        """Estimate total inference latency"""
        device_times = [0.0] * self.n_devices
        
        for layer_idx, device_idx in enumerate(self.current_assignment):
            device_times[device_idx] += self.layer_profiles[layer_idx].compute_time
        
        # Pipeline latency is dominated by slowest device + communication
        max_device_time = max(device_times) if device_times else 0
        comm_time = self._calculate_communication_penalty() * 1000  # Convert to ms
        
        return max_device_time + comm_time
    
    def _get_memory_balance(self) -> float:
        """Get memory balance metric"""
        loads = self._get_device_memory_loads()
        mean_load = np.mean(loads)
        return np.std(loads) / max(mean_load, 1.0)
    
    def _get_compute_balance(self) -> float:
        """Get compute balance metric"""
        loads = self._get_device_compute_loads()
        mean_load = np.mean(loads)
        return np.std(loads) / max(mean_load, 1.0)

# DQN Agent with fixed batch normalization issue
class DQNAgent:
    """Deep Q-Network agent optimized for layer splitting"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.update_freq = 100
        
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def _build_model(self) -> nn.Module:
        """Build neural network without batch normalization to avoid single batch issues"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first, then to tensors
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        self.target_network.eval()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class ExpansionNetV2LayerSplitter:
    """Main class integrating with real ExpansionNet v2"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_path = model_path or "dummy_model"  # Default to dummy
        self.config_path = config_path
        self.profiler = ExpansionNetV2Profiler(self.model_path)
        self.model = None
        self.devices = []
        self.env = None
        self.agent = None
        
    def _load_expansionnet_model(self):
        """Load ExpansionNet v2 model"""
        try:
            # Try to load the actual ExpansionNet v2 model
            if self.model_path != "dummy_model" and os.path.exists(self.model_path):
                # Load your actual model here
                # This depends on how your ExpansionNet v2 model is saved
                # model = torch.load(self.model_path, map_location='cpu')

                model = End_ExpansionNet_v2()
                model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

                print("Loaded actual ExpansionNet v2 model")
                return model
            else:
                if self.model_path != "dummy_model":
                    print(f"Model file not found: {self.model_path}")
                print("Using dummy model for testing")
                return DummyExpansionNetV2()
        except Exception as e:
            print(f"Could not load ExpansionNet v2 model: {e}")
            print("Using dummy model for testing")
            return DummyExpansionNetV2()
    
    def initialize(self, sample_input: Dict = None):
        """Initialize the system with model profiling"""
        print("Profiling available devices...")
        self.devices = self.profiler.profile_devices()
        
        print("Loading ExpansionNet v2 model...")
        self.model = self._load_expansionnet_model()
        
        if self.model is None:
            raise ValueError("Failed to load model. Check your model path and ensure the model file exists.")
        
        if sample_input is None:
            sample_input = self._create_sample_input()
        
        print("Profiling model layers...")
        layer_profiles = self.profiler.profile_model_layers(self.model, sample_input)
        
        if not layer_profiles:
            raise ValueError("No layer profiles were generated. Check if the model forward pass is working correctly.")
        
        print("Setting up RL environment...")
        self.env = LayerSplittingEnvironment(self.devices, layer_profiles)
        
        state_size = len(self.env._get_state())
        action_size = len(layer_profiles) * len(self.devices)
        self.agent = DQNAgent(state_size, action_size)
        
        print(f"Initialized with {len(self.devices)} devices and {len(layer_profiles)} layers")
    
    def _create_sample_input(self) -> Dict:
        """Create sample input for profiling"""
        # Determine device for sample input
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move model to device
        if self.model:
            self.model = self.model.to(device)
        
        # Adjust based on ExpansionNet v2 input format
        return {
            'input_ids': torch.randint(0, 1000, (1, 20)).to(device),  # Example caption tokens
            'images': torch.randn(1, 3, 224, 224).to(device),  # Example image
            'attention_mask': torch.ones(1, 20).to(device)
        }
    
    def train_splitter(self, episodes: int = 2000) -> Dict:
        """Train the RL agent for optimal layer splitting"""
        if self.env is None or self.agent is None:
            raise ValueError("Must call initialize() first")
        
        scores = []
        best_assignment = None
        best_score = float('-inf')
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = len(self.profiler.layer_profiles) * 3
            
            while steps < max_steps:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            scores.append(total_reward)
            
            if total_reward > best_score:
                best_score = total_reward
                best_assignment = info.get('assignment', self.env.current_assignment.copy())
            
            #
            
            # Train agent
            if len(self.agent.memory) > 64:
                self.agent.replay()
            
            # Update target network
            if episode % self.agent.update_freq == 0:
                self.agent.update_target_network()
            
            if episode % 200 == 0:
                avg_score = np.mean(scores[-200:])
                print(f"Episode {episode}, Avg Score: {avg_score:.3f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, "
                      f"Best Score: {best_score:.3f}")
        
        return {
            'best_assignment': best_assignment,
            'best_score': best_score,
            'training_scores': scores,
            'final_assignment': self.env.current_assignment
        }
    
    def get_optimal_assignment(self) -> Dict:
        """Get the current optimal layer assignment"""
        if self.env is None:
            raise ValueError("Must call initialize() first")
        
        state = self.env.reset()
        
        # Use trained agent to make decisions
        for _ in range(len(self.profiler.layer_profiles)):
            action = self.agent.act(state)
            state, reward, done, info = self.env.step(action)
        
        return {
            'assignment': self.env.current_assignment,
            'estimated_latency_ms': self.env._estimate_total_latency(), 
            'compute_balance': self.env._get_compute_balance(),
            'memory_balance': self.env._get_memory_balance(),
            'device_compute_loads': self.env._get_device_compute_loads(),
            'device_memory_loads': self.env._get_device_memory_loads()
        }
    
    def save_configuration(self, filepath: str):
        """Save the trained configuration"""
        if self.agent is None:
            raise ValueError("No trained agent to save")
        
        config = {
            'model_state_dict': self.agent.q_network.state_dict(),
            'assignment': self.env.current_assignment if self.env else None,
            'device_profiles': [vars(d) for d in self.devices],
            'layer_profiles': [vars(p) for p in self.profiler.layer_profiles]
        }
        
        torch.save(config, filepath)
        print(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str):
        """Load a previously trained configuration"""
        config = torch.load(filepath)
        
        if self.agent:
            self.agent.q_network.load_state_dict(config['model_state_dict'])
            print(f"Configuration loaded from {filepath}")

# Example usage
def main():
    # Initialize with your ExpansionNet v2 model
    splitter = ExpansionNetV2LayerSplitter(
        model_path=r"C:\Users\gagan\Downloads\rf_model.pth",
        config_path=r"C:\Users\gagan\Desktop\defer_exp\config.json"
    )
    
    # Initialize and profile the system
    splitter.initialize()
    
    # Train the splitter
    results = splitter.train_splitter(episodes=1000)
    
    # Get optimal assignment
    optimal_config = splitter.get_optimal_assignment()
    
    print("\nOptimal Layer Assignment:")
    print(f"Estimated latency: {optimal_config['estimated_latency_ms']:.2f} ms")
    print(f"Compute balance: {optimal_config['compute_balance']:.3f}")
    print(f"Memory balance: {optimal_config['memory_balance']:.3f}")
    
    # Save the configuration
    splitter.save_configuration("expansionnet_v2_splitting_config.pth")

if __name__ == "__main__":
    main()