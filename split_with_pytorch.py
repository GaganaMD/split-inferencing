import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import time

class YOLOv5InferenceNode:
    """
    Represents a single compute node for distributed YOLOv5 inference
    """
    def __init__(self, node_id: int, layers: List[nn.Module], layer_names: List[str], 
                 device: str = 'cpu'):
        self.node_id = node_id
        self.layers = nn.ModuleList(layers)
        self.layer_names = layer_names
        self.device = device
        self.layers.to(device)
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        print(f"✓ Node {node_id} initialized with {len(layers)} layers on {device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this node's layers
        """
        start_time = time.time()
        
        x = x.to(self.device)
        
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x)
            except Exception as e:
                print(f"Error in node {self.node_id}, layer {i} ({self.layer_names[i]}): {e}")
                raise e
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        return x
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this node
        """
        if not self.inference_times:
            return {"avg_time": 0, "total_inferences": 0}
        
        return {
            "node_id": self.node_id,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "total_inferences": self.total_inferences,
            "num_layers": len(self.layers)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times = []
        self.total_inferences = 0

class DistributedYOLOv5:
    """
    Distributed YOLOv5 inference system
    """
    def __init__(self, model, distribution: Dict[str, Dict], devices: List[str] = None):
        self.original_model = model
        self.distribution = distribution
        self.devices = devices or ['cpu'] * len(distribution)
        self.nodes = []
        
        # Extract layers from original model
        self.all_layers, self.all_layer_names = self._extract_all_layers(model)
        
        # Create inference nodes
        self._create_inference_nodes()
        
        print(f"✓ Distributed YOLOv5 system created with {len(self.nodes)} nodes")
    
    def _extract_all_layers(self, model):
        """Extract all layers from the model in order"""
        layers = []
        layer_names = []
        
        # Handle different model types
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        def extract_layers(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if len(list(child.children())) > 0:
                    extract_layers(child, full_name)
                else:
                    layers.append(child)
                    layer_names.append(full_name)
        
        extract_layers(actual_model)
        return layers, layer_names
    
    def _create_inference_nodes(self):
        """Create inference nodes based on distribution"""
        for i, (node_name, node_info) in enumerate(self.distribution.items()):
            device = self.devices[i] if i < len(self.devices) else 'cpu'
            
            # Get layers for this node
            layer_indices = node_info['layer_indices']
            node_layers = [self.all_layers[idx] for idx in layer_indices]
            node_layer_names = [self.all_layer_names[idx] for idx in layer_indices]
            
            # Create node
            node = YOLOv5InferenceNode(
                node_id=i,
                layers=node_layers,
                layer_names=node_layer_names,
                device=device
            )
            
            self.nodes.append(node)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed forward pass through all nodes
        """
        print(f"Starting distributed inference with input shape: {x.shape}")
        total_start_time = time.time()
        
        current_output = x
        
        for i, node in enumerate(self.nodes):
            print(f"  Processing on node {i} ({node.device})...")
            node_start_time = time.time()
            
            current_output = node.forward(current_output)
            
            node_time = time.time() - node_start_time
            print(f"  Node {i} completed in {node_time:.4f}s, output shape: {current_output.shape}")
        
        total_time = time.time() - total_start_time
        print(f"✓ Distributed inference completed in {total_time:.4f}s")
        
        return current_output
    
    def benchmark(self, input_tensor: torch.Tensor, num_runs: int = 5):
        """
        Benchmark the distributed inference system
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking Distributed YOLOv5 ({num_runs} runs)")
        print(f"{'='*60}")
        
        # Reset stats
        for node in self.nodes:
            node.reset_stats()
        
        total_times = []
        
        # Warm up
        print("Warming up...")
        _ = self.forward(input_tensor)
        
        # Benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        for i in range(num_runs):
            start_time = time.time()
            _ = self.forward(input_tensor)
            total_time = time.time() - start_time
            total_times.append(total_time)
            print(f"Run {i+1}: {total_time:.4f}s")
        
        # Print results
        self._print_benchmark_results(total_times)
    
    def _print_benchmark_results(self, total_times: List[float]):
        """Print detailed benchmark results"""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        # Overall performance
        print(f"Overall Performance:")
        print(f"  Average total time: {np.mean(total_times):.4f}s")
        print(f"  Min total time: {np.min(total_times):.4f}s")
        print(f"  Max total time: {np.max(total_times):.4f}s")
        print(f"  Std deviation: {np.std(total_times):.4f}s")
        print()
        
        # Per-node performance
        print("Per-Node Performance:")
        for node in self.nodes:
            stats = node.get_performance_stats()
            print(f"  Node {stats['node_id']} ({node.device}):")
            print(f"    Layers: {stats['num_layers']}")
            print(f"    Avg time: {stats['avg_inference_time']:.4f}s")
            print(f"    Min time: {stats['min_inference_time']:.4f}s")
            print(f"    Max time: {stats['max_inference_time']:.4f}s")
        print()
        
        # Load balance analysis
        node_times = [np.mean(node.inference_times) for node in self.nodes]
        max_time = max(node_times)
        min_time = min(node_times)
        load_imbalance = ((max_time - min_time) / max_time) * 100
        
        print(f"Load Balance Analysis:")
        print(f"  Load imbalance: {load_imbalance:.1f}%")
        print(f"  Bottleneck node: Node {node_times.index(max_time)} ({max_time:.4f}s)")
        print(f"  Fastest node: Node {node_times.index(min_time)} ({min_time:.4f}s)")

def create_test_input(batch_size: int = 1, img_size: int = 640) -> torch.Tensor:
    """Create test input tensor for YOLOv5"""
    return torch.randn(batch_size, 3, img_size, img_size)

def main_distributed_inference():
    """
    Main function to demonstrate distributed YOLOv5 inference
    """
    print("YOLOv5 Distributed Inference Demo")
    print("="*60)
    
    try:
        # Load YOLOv5 model
        print("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        print("✓ Model loaded successfully")
        
        # Create simple sequential distribution for 4 nodes
        # (You can replace this with the smart distribution from the previous code)
        print("\nAnalyzing model architecture...")
        
        # Extract layers
        layers = []
        layer_names = []
        
        def extract_layers(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if len(list(child.children())) > 0:
                    extract_layers(child, full_name)
                else:
                    layers.append(child)
                    layer_names.append(full_name)
        
        extract_layers(model.model)
        
        total_layers = len(layers)
        n_nodes = 4
        layers_per_node = total_layers // n_nodes
        
        print(f"Total layers: {total_layers}")
        print(f"Distributing across {n_nodes} nodes")
        
        # Create distribution
        distribution = {}
        for i in range(n_nodes):
            start_idx = i * layers_per_node
            if i == n_nodes - 1:  # Last node gets remaining layers
                end_idx = total_layers
            else:
                end_idx = (i + 1) * layers_per_node
            
            distribution[f'node_{i}'] = {
                'layer_indices': list(range(start_idx, end_idx)),
                'num_layers': end_idx - start_idx
            }
        
        # Print distribution
        print("\nLayer distribution:")
        for node_name, node_info in distribution.items():
            print(f"  {node_name}: {node_info['num_layers']} layers "
                  f"(indices {node_info['layer_indices'][0]}-{node_info['layer_indices'][-1]})")
        
        # Create distributed system
        print("\nCreating distributed inference system...")
        devices = ['cpu'] * n_nodes  # Use ['cuda:0', 'cuda:1', ...] if you have multiple GPUs
        
        distributed_yolo = DistributedYOLOv5(model, distribution, devices)
        
        # Create test input
        print("\nCreating test input...")
        test_input = create_test_input(batch_size=1, img_size=640)
        print(f"Test input shape: {test_input.shape}")
        
        # Run inference
        print("\nRunning distributed inference...")
        output = distributed_yolo.forward(test_input)
        print(f"Final output shape: {output.shape}")
        
        # Benchmark
        print("\nRunning benchmark...")
        distributed_yolo.benchmark(test_input, num_runs=3)
        
        print("\n✓ Distributed inference demo completed successfully!")
        
        return distributed_yolo
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. torch and torchvision installed")
        print("2. Internet connection for downloading YOLOv5")
        return None

if __name__ == "__main__":
    distributed_system = main_distributed_inference()