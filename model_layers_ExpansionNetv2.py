import torch
import os
import sys
from collections import Counter

# Add ExpansionNet_v2 to the Python path (relative to this script)
expansionnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ExpansionNet_v2"))
sys.path.append(expansionnet_path)

# VSCode Pylance will recognize this if you also add the path in .vscode/settings.json
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2


# Handle __file__ fallback for Jupyter or interactive environments
try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    script_name = "expansionnetv2_analysis"

# Create output directory
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
file_path = os.path.join(result_dir, f"{script_name}.txt")

def load_expansionnetv2_model():
    """
    Load ExpansionNetv2 model with proper error handling and instructions
    """
    # Check if ExpansionNet_v2 exists
    if os.path.exists(expansionnet_path):
        try:
            # Instantiate model
            model = End_ExpansionNet_v2(
                swin_img_size=384,
                swin_patch_size=4,
                swin_in_chans=3,
                swin_embed_dim=192,
                swin_depths=[2, 2, 18, 2],
                swin_num_heads=[6, 12, 24, 48],
                swin_window_size=12,
                swin_mlp_ratio=4.,
                swin_qkv_bias=True,
                swin_qk_scale=None,
                swin_drop_rate=0.,
                swin_attn_drop_rate=0.,
                swin_drop_path_rate=0.2,
                swin_norm_layer=torch.nn.LayerNorm,
                swin_ape=False,
                swin_patch_norm=True,
                swin_use_checkpoint=False,
                final_swin_dim=1536,
                d_model=512,
                N_enc=3,
                N_dec=3,
                num_heads=8,
                ff=2048,
                num_exp_enc_list=[32, 64, 128],
                num_exp_dec=16,
                output_word2idx={"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3},
                output_idx2word={0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"},
                max_seq_len=74,
                drop_args={'enc': 0.0, 'dec': 0.0, 'enc_input': 0.0, 'dec_input': 0.0, 'other': 0.0},
                rank=0
            )
            print("✓ Successfully loaded ExpansionNetv2 model from local directory")
            return model, "local_directory"
        except Exception as e:
            print(f"✗ Error importing from local directory: {e}")
            return None, f"import_error: {e}"
    else:
        print("✗ ExpansionNet_v2 directory not found")
        return None, "directory_not_found"

def count_layer_types(module):
    return Counter(type(layer).__name__ for layer in module.modules())

def get_layer_details(module, prefix="", max_depth=3, current_depth=0):
    details = []
    if current_depth >= max_depth:
        return details
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        layer_type = type(layer).__name__
        details.append((full_name, layer_type))
        if list(layer.children()) and current_depth < max_depth - 1:
            details.extend(get_layer_details(layer, full_name, max_depth, current_depth + 1))
    return details

def analyze_expansionnet_components(model):
    components = {}
    component_names = ['swin_transf', 'enc_reduce_group', 'enc', 'dec',
                       'out_embedder', 'pos_embedder', 'enc_embedder']
    for comp_name in component_names:
        if hasattr(model, comp_name):
            component = getattr(model, comp_name)
            components[comp_name] = count_layer_types(component)
    return components

# Try to load the model
print("Attempting to load ExpansionNetv2...")
model, load_status = load_expansionnetv2_model()

if model is None:
    with open(file_path, "w") as f:
        f.write("ExpansionNetv2 Analysis - SETUP REQUIRED\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Load Status: {load_status}\n\n")
        f.write("SETUP INSTRUCTIONS:\n")
        f.write("1. Clone the repository:\n")
        f.write("   git clone https://github.com/jchenghu/ExpansionNet_v2.git\n\n")
        f.write("2. Ensure ExpansionNet_v2 is in the same directory as this script\n")
        f.write("3. Re-run this script\n")
    print(f"Setup instructions written to: {file_path}")
    sys.exit(1)

# Analyze model
print("Analyzing model architecture...")
freq_model = count_layer_types(model)
component_frequencies = analyze_expansionnet_components(model)
layer_details = get_layer_details(model, max_depth=3)

with open(file_path, "w") as f:
    f.write(f"ExpansionNetv2 Architecture Analysis\n")
    f.write("=" * 50 + "\n")
    f.write(f"Load Status: {load_status}\n")
    f.write(f"Model Type: {type(model).__name__}\n")
    f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n\n")

    f.write("OVERALL LAYER TYPE FREQUENCIES:\n")
    for layer_type, count in sorted(freq_model.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  {layer_type}: {count}\n")

    if component_frequencies:
        f.write("\nCOMPONENT-SPECIFIC ANALYSIS:\n")
        for component_name, freq_dict in component_frequencies.items():
            f.write(f"\n{component_name.upper()}:\n")
            for layer_type, count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {layer_type}: {count}\n")

    f.write("\n" + "=" * 50 + "\n")
    f.write("DETAILED LAYER STRUCTURE (Top 3 levels):\n")
    for layer_name, layer_type in layer_details[:100]:
        f.write(f"{layer_name}: {layer_type}\n")
    if len(layer_details) > 100:
        f.write(f"\n... and {len(layer_details) - 100} more layers\n")

    f.write(f"\nSUMMARY STATISTICS:\n")
    f.write(f"Total unique layer types: {len(freq_model)}\n")
    f.write(f"Total module instances: {sum(freq_model.values())}\n")
    f.write(f"Named modules analyzed: {len(layer_details)}\n")
    f.write(f"\nTop 10 most common layer types:\n")
    for i, (layer_type, count) in enumerate(freq_model.most_common(10), 1):
        f.write(f"  {i}. {layer_type}: {count}\n")

print(f"✓ ExpansionNetv2 analysis completed!")
print(f"Results saved to: {file_path}")
print(f"Model successfully loaded via: {load_status}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Total layer types: {len(freq_model)}")
print("Most common layer types:")
for layer_type, count in freq_model.most_common(5):
    print(f"  {layer_type}: {count}")


# import torch
# import os
# import sys
# from collections import Counter

# sys.path.append(os.path.abspath('ExpansionNet_v2'))

# from models import End_ExpansionNet_v2
# from models.End_ExpansionNet_v2 import YourModelClass



# # Handle __file__ fallback for Jupyter or interactive environments
# try:
#     script_name = os.path.splitext(os.path.basename(__file__))[0]
# except NameError:
#     script_name = "expansionnetv2_analysis"

# # Create output directory
# result_dir = "result"
# os.makedirs(result_dir, exist_ok=True)
# file_path = os.path.join(result_dir, f"{script_name}.txt")

# def load_expansionnetv2_model():
#     """
#     Load ExpansionNetv2 model with proper error handling and instructions
#     """
#     # First check if the ExpansionNet_v2_src directory exists
#     expansion_net_path = "./ExpansionNet_v2_src"
#     if os.path.exists(expansion_net_path):
#         # Add the path to sys.path so we can import the modules
#         sys.path.insert(0, expansion_net_path)
#         try:
#             # Try to import the model class
#             from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
            
#             # Create model instance with default parameters
#             # These are typical parameters based on the repository
#             model = End_ExpansionNet_v2(
#                 swin_img_size=384,
#                 swin_patch_size=4,
#                 swin_in_chans=3,
#                 swin_embed_dim=192,
#                 swin_depths=[2, 2, 18, 2],
#                 swin_num_heads=[6, 12, 24, 48],
#                 swin_window_size=12,
#                 swin_mlp_ratio=4.,
#                 swin_qkv_bias=True,
#                 swin_qk_scale=None,
#                 swin_drop_rate=0.,
#                 swin_attn_drop_rate=0.,
#                 swin_drop_path_rate=0.2,
#                 swin_norm_layer=torch.nn.LayerNorm,
#                 swin_ape=False,
#                 swin_patch_norm=True,
#                 swin_use_checkpoint=False,
#                 final_swin_dim=1536,
#                 d_model=512,
#                 N_enc=3,
#                 N_dec=3,
#                 num_heads=8,
#                 ff=2048,
#                 num_exp_enc_list=[32, 64, 128],
#                 num_exp_dec=16,
#                 output_word2idx={"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3},  # Minimal vocab
#                 output_idx2word={0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"},
#                 max_seq_len=74,
#                 drop_args={
#                     'enc': 0.0, 'dec': 0.0, 'enc_input': 0.0, 'dec_input': 0.0, 'other': 0.0
#                 },
#                 rank=0
#             )
            
#             print("✓ Successfully loaded ExpansionNetv2 model from local directory")
#             return model, "local_directory"
            
#         except Exception as e:
#             print(f"✗ Error importing from local directory: {e}")
#             return None, f"import_error: {e}"
    
#     else:
#         print("✗ ExpansionNet_v2_src directory not found")
#         return None, "directory_not_found"

# def count_layer_types(module):
#     """Count the frequency of each layer type in a module"""
#     return Counter(type(layer).__name__ for layer in module.modules())

# def get_layer_details(module, prefix="", max_depth=3, current_depth=0):
#     """Get detailed layer names and types with depth limiting"""
#     details = []
#     if current_depth >= max_depth:
#         return details
        
#     for name, layer in module.named_children():
#         full_name = f"{prefix}.{name}" if prefix else name
#         layer_type = type(layer).__name__
#         details.append((full_name, layer_type))
#         # Recursively get children
#         if list(layer.children()) and current_depth < max_depth - 1:
#             details.extend(get_layer_details(layer, full_name, max_depth, current_depth + 1))
#     return details

# def analyze_expansionnet_components(model):
#     """Analyze different components of ExpansionNetv2"""
#     components = {}
    
#     # Common ExpansionNet components
#     component_names = ['swin_transf', 'enc_reduce_group', 'enc', 'dec', 
#                       'out_embedder', 'pos_embedder', 'enc_embedder']
    
#     for comp_name in component_names:
#         if hasattr(model, comp_name):
#             component = getattr(model, comp_name)
#             components[comp_name] = count_layer_types(component)
    
#     return components

# # Try to load the model
# print("Attempting to load ExpansionNetv2...")
# model, load_status = load_expansionnetv2_model()

# if model is None:
#     # Write instructions to file if model couldn't be loaded
#     with open(file_path, "w") as f:
#         f.write("ExpansionNetv2 Analysis - SETUP REQUIRED\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"Load Status: {load_status}\n\n")
#         f.write("SETUP INSTRUCTIONS:\n")
#         f.write("1. Clone the repository:\n")
#         f.write("   git clone https://github.com/jchenghu/ExpansionNet_v2.git\n\n")
#         f.write("2. Navigate to the cloned directory:\n")
#         f.write("   cd ExpansionNet_v2\n\n")
#         f.write("3. Install requirements:\n")
#         f.write("   pip install -r requirements.txt\n\n")
#         f.write("4. Move this script to the ExpansionNet_v2 directory\n")
#         f.write("   or ensure ExpansionNet_v2_src is in the same directory as this script\n\n")
#         f.write("5. Re-run this script\n\n")
#         f.write("The ExpansionNet_v2_src directory should contain:\n")
#         f.write("- models/End_ExpansionNet_v2.py\n")
#         f.write("- Other necessary modules\n\n")
#         f.write("Alternative: You can also download the pretrained model from:\n")
#         f.write("https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3\n")
    
#     print(f"Setup instructions written to: {file_path}")
#     print("\nTo analyze ExpansionNetv2, you need to:")
#     print("1. git clone https://github.com/jchenghu/ExpansionNet_v2.git")
#     print("2. pip install -r ExpansionNet_v2/requirements.txt")
#     print("3. Ensure ExpansionNet_v2_src is accessible from this script")
#     sys.exit(1)

# # If model loaded successfully, perform analysis
# print("Analyzing model architecture...")

# # Collect frequencies for different levels
# freq_model = count_layer_types(model)

# # Analyze specific components
# component_frequencies = analyze_expansionnet_components(model)

# # Get detailed layer structure (limited depth to avoid overwhelming output)
# layer_details = get_layer_details(model, max_depth=3)

# # Write comprehensive analysis to file
# with open(file_path, "w") as f:
#     f.write(f"ExpansionNetv2 Architecture Analysis\n")
#     f.write("=" * 50 + "\n")
#     f.write(f"Load Status: {load_status}\n")
#     f.write(f"Model Type: {type(model).__name__}\n")
#     f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
#     f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n\n")
    
#     f.write("OVERALL LAYER TYPE FREQUENCIES:\n")
#     f.write("-" * 30 + "\n")
#     for layer_type, count in sorted(freq_model.items(), key=lambda x: x[1], reverse=True):
#         f.write(f"  {layer_type}: {count}\n")
    
#     # Write component-specific frequencies
#     if component_frequencies:
#         f.write("\nCOMPONENT-SPECIFIC ANALYSIS:\n")
#         f.write("-" * 30 + "\n")
#         for component_name, freq_dict in component_frequencies.items():
#             f.write(f"\n{component_name.upper()}:\n")
#             for layer_type, count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
#                 f.write(f"  {layer_type}: {count}\n")
    
#     f.write("\n" + "=" * 50 + "\n")
#     f.write("DETAILED LAYER STRUCTURE (Top 3 levels):\n")
#     f.write("=" * 50 + "\n")
#     for layer_name, layer_type in layer_details[:100]:  # Limit output
#         f.write(f"{layer_name}: {layer_type}\n")
    
#     if len(layer_details) > 100:
#         f.write(f"\n... and {len(layer_details) - 100} more layers\n")
    
#     f.write(f"\nSUMMARY STATISTICS:\n")
#     f.write("-" * 20 + "\n")
#     f.write(f"Total unique layer types: {len(freq_model)}\n")
#     f.write(f"Total module instances: {sum(freq_model.values())}\n")
#     f.write(f"Named modules analyzed: {len(layer_details)}\n")
    
#     # Most common layer types
#     f.write(f"\nTop 10 most common layer types:\n")
#     for i, (layer_type, count) in enumerate(freq_model.most_common(10), 1):
#         f.write(f"  {i}. {layer_type}: {count}\n")

# print(f"✓ ExpansionNetv2 analysis completed!")
# print(f"Results saved to: {file_path}")
# print(f"Model successfully loaded via: {load_status}")
# print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# print(f"Total layer types: {len(freq_model)}")
# print(f"Most common layer types:")
# for layer_type, count in freq_model.most_common(5):
#     print(f"  {layer_type}: {count}")