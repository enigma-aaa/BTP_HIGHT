#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('/mnt/hdd/gtoken/data1/ayush/HIGHT_4')

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.datasets.smiles2graph import smiles2graph
from llava.model.multimodal_encoder.builder import build_graph_tower
import json

def test_gpm_generation():
    disable_torch_init()
    
    # Load model
    model_path = "./checkpoints/Graph-LLaVA-4C-gpm-5ep-hlinear-fgprompt-neg-extend/graph-text-molgen/property_pred-llava-gpm-lmsys/vicuna-7b-v1.3-finetune_lora-5ep"
    model_base = "lmsys/vicuna-7b-v1.3"
    model_name = get_model_name_from_path(model_base)
    
    print(f"Loading model from: {model_path}")
    print(f"Using base model: {model_base}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"
    )
    
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {[attr for attr in dir(model) if 'prepare' in attr.lower()]}")
    
    # Check if this is actually a LLaVA model
    if hasattr(model, 'model') and hasattr(model.model, 'prepare_inputs_labels_for_multimodal'):
        print("✅ Model has multimodal capabilities")
    else:
        print("❌ Model does not have multimodal capabilities")
        print("Trying to manually create LLaVA model...")
        
        # Try to manually create the LLaVA model
        from llava.model.language_model.llava_graph_llama import LlavaGraphLlamaForCausalLM, LlavaGraphLlamaConfig
        from transformers import AutoConfig
        
        # Load the config
        config = AutoConfig.from_pretrained(model_path)
        print(f"Config model type: {config.model_type}")
        print(f"Config has mm_projector_type: {hasattr(config, 'mm_projector_type')}")
        print(f"Config mm_projector_type: {getattr(config, 'mm_projector_type', 'None')}")
        print(f"Config has mm_hidden_size: {hasattr(config, 'mm_hidden_size')}")
        print(f"Config mm_hidden_size: {getattr(config, 'mm_hidden_size', 'None')}")
        
        # Create the LLaVA model
        llava_model = LlavaGraphLlamaForCausalLM.from_pretrained(model_base, config=config, **{"device_map": "auto"})
        
        print(f"After creation - Model has mm_projector: {hasattr(llava_model.model, 'mm_projector')}")
        if hasattr(llava_model.model, 'mm_projector'):
            print(f"MM projector type: {type(llava_model.model.mm_projector)}")
            print(f"MM projector layers: {len(llava_model.model.mm_projector)}")
        
        # Load the LoRA weights
        from peft import PeftModel
        llava_model = PeftModel.from_pretrained(llava_model, model_path)
        
        print(f"After LoRA loading - Model has mm_projector: {hasattr(llava_model.model, 'mm_projector')}")
        if hasattr(llava_model.model, 'mm_projector'):
            print(f"MM projector type: {type(llava_model.model.mm_projector)}")
            print(f"MM projector layers: {len(llava_model.model.mm_projector)}")
        else:
            print("❌ MM projector lost after LoRA loading!")
            # Check if it's in the base model
            print(f"Base model has mm_projector: {hasattr(llava_model.base_model.model, 'mm_projector')}")
            if hasattr(llava_model.base_model.model, 'mm_projector'):
                print("✅ MM projector is in base model")
                # Copy mm_projector from base model to model
                llava_model.model.mm_projector = llava_model.base_model.model.mm_projector
                print("✅ Copied mm_projector from base model to model")
        
        # Load non-LoRA trainables
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            print(f"Non-LoRA trainables keys: {list(non_lora_trainables.keys())}")
            
            # Check if mm_projector weights are in the non-LoRA trainables
            mm_projector_keys = [k for k in non_lora_trainables.keys() if 'mm_projector' in k]
            print(f"MM projector keys in non-LoRA trainables: {mm_projector_keys}")
            
            if mm_projector_keys:
                # Manually recreate the mm_projector from the weights
                from torch import nn
                mm_projector = nn.ModuleList()
                
                # Get the number of layers from the keys
                layer_count = max([int(k.split('.')[-2]) for k in mm_projector_keys if 'weight' in k]) + 1
                print(f"Recreating mm_projector with {layer_count} layers")
                
                for i in range(layer_count):
                    weight_key = f'base_model.model.model.mm_projector.{i}.weight'
                    bias_key = f'base_model.model.model.mm_projector.{i}.bias'
                    
                    if weight_key in non_lora_trainables and bias_key in non_lora_trainables:
                        weight = non_lora_trainables[weight_key]
                        bias = non_lora_trainables[bias_key]
                        
                        # Create linear layer
                        linear_layer = nn.Linear(weight.shape[1], weight.shape[0])
                        linear_layer.weight.data = weight
                        linear_layer.bias.data = bias
                        mm_projector.append(linear_layer)
                        print(f"✅ Created layer {i}: {weight.shape[1]} -> {weight.shape[0]}")
                
                # Add mm_projector to the model and move to correct device
                llava_model.model.mm_projector = mm_projector
                # Move mm_projector to the same device as the model
                device = next(llava_model.parameters()).device
                llava_model.model.mm_projector = llava_model.model.mm_projector.to(device)
                print(f"✅ Manually created mm_projector with {len(mm_projector)} layers on device {device}")
            
            llava_model.load_state_dict(non_lora_trainables, strict=False)
            print("✅ Loaded non-LoRA trainables")
        else:
            print("❌ No non-LoRA trainables found")
        
        # Check if mm_projector exists after loading
        print(f"After loading - Model has mm_projector: {hasattr(llava_model.model, 'mm_projector')}")
        if hasattr(llava_model.model, 'mm_projector'):
            print(f"MM projector type: {type(llava_model.model.mm_projector)}")
            print(f"MM projector layers: {len(llava_model.model.mm_projector)}")
        
        model = llava_model
        print(f"New model type: {type(model)}")
        
        # Check if the model has the expected components
        print(f"Model has graph tower: {hasattr(model, 'get_graph_tower')}")
        print(f"Model has mm_projector: {hasattr(model.model, 'mm_projector')}")
        if hasattr(model.model, 'mm_projector'):
            print(f"MM projector type: {type(model.model.mm_projector)}")
            print(f"MM projector layers: {len(model.model.mm_projector)}")
    
    # Test with a simple SMILES
    test_smiles = "CCO"  # Ethanol
    print(f"Testing with SMILES: {test_smiles}")
    
    # Convert SMILES to graph
    try:
        graph_data = smiles2graph(test_smiles)
        print(f"Graph data keys: {graph_data.keys()}")
        num_edges = graph_data['edge_index'].shape[1] if 'edge_index' in graph_data else 0
        print(f"Graph has {graph_data['num_nodes']} nodes and {num_edges} edges")
    except Exception as e:
        print(f"Error converting SMILES to graph: {e}")
        return
    
    # Test conversation
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    # Simple test question
    question = "What is the HOMO energy of this molecule?"
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"Prompt: {prompt}")
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    
    # Process input through multimodal pipeline
    print("Processing input through multimodal pipeline...")
    try:
        # Convert graph data to the format expected by the model
        from torch_geometric.data import Data
        graph_data_obj = Data(
            x=torch.tensor(graph_data['node_feat'], dtype=torch.float32),
            edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph_data['edge_feat'], dtype=torch.float32),
            num_nodes=graph_data['num_nodes'],
            lap_pe=torch.tensor(graph_data['lap_pe'], dtype=torch.float32)
        )
        
        # Add num_part attribute for hlinear projector
        # num_part[0] = number of nodes, num_part[1] = number of motifs (0 for simple graphs)
        graph_data_obj.num_part = torch.tensor([graph_data['num_nodes'], 0], dtype=torch.long)
        
        # Check graph data for NaN values before processing
        print(f"Graph data x mean: {graph_data_obj.x.mean()}, std: {graph_data_obj.x.std()}")
        print(f"Graph data edge_attr mean: {graph_data_obj.edge_attr.mean()}, std: {graph_data_obj.edge_attr.std()}")
        print(f"Graph data lap_pe mean: {graph_data_obj.lap_pe.mean()}, std: {graph_data_obj.lap_pe.std()}")
        
        if torch.isnan(graph_data_obj.x.mean()) or torch.isinf(graph_data_obj.x.mean()):
            print("❌ Graph data x contains NaN or Inf values!")
            return
        if torch.isnan(graph_data_obj.edge_attr.mean()) or torch.isinf(graph_data_obj.edge_attr.mean()):
            print("❌ Graph data edge_attr contains NaN or Inf values!")
            return
        if torch.isnan(graph_data_obj.lap_pe.mean()) or torch.isinf(graph_data_obj.lap_pe.mean()):
            print("❌ Graph data lap_pe contains NaN or Inf values!")
            return
        
        print("✅ Graph data looks normal")
        
        # Test the graph tower directly
        print("Testing graph tower directly...")
        try:
            graph_tower = model.get_graph_tower()
            print(f"Graph tower type: {type(graph_tower)}")
            
            # Check GPM model parameters
            print("Checking GPM model parameters...")
            gpm_model = graph_tower.model
            print(f"GPM model type: {type(gpm_model)}")
            
            if gpm_model is None:
                print("❌ GPM model is None! This means the model creation failed.")
                print("Let's try to create the model manually...")
                
                # Try to create the model manually
                try:
                    from GPM.model.model import Model as GPMModel
                    
                    # Prepare params for this batch
                    params = dict(graph_tower.params_template)
                    params['device'] = graph_data_obj.x.device
                    params['input_dim'] = graph_data_obj.x.size(1)
                    params['edge_dim'] = 0  # Disable edge features
                    
                    print(f"Creating GPM model with params: {params}")
                    gpm_model = GPMModel(params=params)
                    print(f"✅ GPM model created successfully: {type(gpm_model)}")
                    
                    # Update the graph tower's model
                    graph_tower.model = gpm_model
                    
                except Exception as e:
                    print(f"❌ Failed to create GPM model: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            
            # Check if GPM model has the expected components
            print(f"GPM model has pattern_encoder: {hasattr(gpm_model, 'pattern_encoder')}")
            print(f"GPM model has vq: {hasattr(gpm_model, 'vq')}")
            print(f"GPM model has encoder: {hasattr(gpm_model, 'encoder')}")
            
            # Check some key parameters
            if hasattr(gpm_model, 'pattern_encoder'):
                print(f"Pattern encoder type: {type(gpm_model.pattern_encoder)}")
                if hasattr(gpm_model.pattern_encoder, 'atom_encoder'):
                    print(f"Atom encoder type: {type(gpm_model.pattern_encoder.atom_encoder)}")
                    print(f"Atom encoder weight mean: {gpm_model.pattern_encoder.atom_encoder.linear.weight.mean()}")
                    print(f"Atom encoder weight std: {gpm_model.pattern_encoder.atom_encoder.linear.weight.std()}")
                    if torch.isnan(gpm_model.pattern_encoder.atom_encoder.linear.weight.mean()):
                        print("❌ Atom encoder weights contain NaN!")
                    else:
                        print("✅ Atom encoder weights look normal")
            
            # Test the graph tower directly first
            print("Testing graph tower directly...")
            try:
                graph_features = graph_tower(graph_data_obj)
                if isinstance(graph_features, tuple):
                    graph_features = graph_features[0]
                print(f"Graph features shape: {graph_features.shape}")
                print(f"Graph features mean: {graph_features.mean()}, std: {graph_features.std()}")
                
                if torch.isnan(graph_features.mean()) or torch.isinf(graph_features.mean()):
                    print("❌ Graph tower produces NaN or Inf values!")
                    return
                else:
                    print("✅ Graph tower produces normal values")
                    return  # Success!
            except Exception as e:
                print(f"❌ Graph tower test failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Test the graph tower forward pass step by step
            print("Testing GPM model forward pass step by step...")
            
            # Test pattern generation first
            try:
                from GPM.model.random_walk import get_patterns
                patterns, eids = get_patterns(graph_data_obj, params)
                print(f"Patterns shape: {patterns.shape}")
                print(f"Eids shape: {eids.shape}")
                print(f"Patterns dtype: {patterns.dtype}")
                print(f"Eids dtype: {eids.dtype}")
                
                # Check for valid patterns (no negative indices)
                if (patterns < 0).any():
                    print("❌ Patterns contain negative indices!")
                    return
                if (eids < 0).any():
                    print("❌ Eids contain negative indices!")
                    return
                
                print(f"Patterns min: {patterns.min()}, max: {patterns.max()}")
                print(f"Eids min: {eids.min()}, max: {eids.max()}")
                
                # Add pattern_set to params
                params['pattern_set'] = {
                    'pattern': patterns,
                    'eid': eids,
                }
                print("✅ Added pattern_set to params")
                
                print("✅ Pattern generation looks normal")
                
                # Test the GPM model directly
                print("Testing GPM model forward pass directly...")
                try:
                    # Move patterns and eids to the correct device
                    device = graph_data_obj.x.device
                    patterns = patterns.to(device)
                    eids = eids.to(device)
                    
                    # Test the GPM model's encode_node method
                    print("Testing GPM model encode_node...")
                    # Select all nodes for encoding
                    num_nodes = graph_data_obj.x.size(0)
                    nodes = torch.arange(num_nodes, device=device)
                    pred, node_emb, pattern_emb, commit_loss = gpm_model.encode_node(graph_data_obj, nodes, params, mode='inference')
                    print(f"Node embeddings shape: {node_emb.shape}")
                    print(f"Node embeddings mean: {node_emb.mean()}, std: {node_emb.std()}")
                    
                    if torch.isnan(node_emb.mean()) or torch.isinf(node_emb.mean()):
                        print("❌ GPM model encode_node produces NaN or Inf values!")
                        print(f"Pred shape: {pred.shape}, mean: {pred.mean()}, std: {pred.std()}")
                        print(f"Pattern emb shape: {pattern_emb.shape}, mean: {pattern_emb.mean()}, std: {pattern_emb.std()}")
                        print(f"Commit loss: {commit_loss}")
                        return
                    else:
                        print("✅ GPM model encode_node looks normal")
                    
                except Exception as e:
                    print(f"❌ GPM model forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                
            except Exception as e:
                print(f"❌ Pattern generation failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Test the graph tower forward pass
            graph_output = graph_tower(graph_data_obj)
            print(f"Graph tower output type: {type(graph_output)}")
            print(f"Graph tower output: {graph_output}")
            
            # Handle tuple output
            if isinstance(graph_output, tuple):
                print(f"Graph tower returns tuple with {len(graph_output)} elements")
                for i, output in enumerate(graph_output):
                    print(f"Output {i} type: {type(output)}")
                    if hasattr(output, 'shape'):
                        print(f"Output {i} shape: {output.shape}")
                        if hasattr(output, 'mean'):
                            print(f"Output {i} mean: {output.mean()}, std: {output.std()}")
                            if torch.isnan(output.mean()) or torch.isinf(output.mean()):
                                print(f"❌ Graph tower output {i} contains NaN or Inf values!")
                                return
                # Use the first output as graph features
                graph_features = graph_output[0]
            else:
                graph_features = graph_output
            
            print(f"Graph features shape: {graph_features.shape}")
            print(f"Graph features mean: {graph_features.mean()}, std: {graph_features.std()}")
            
            if torch.isnan(graph_features.mean()) or torch.isinf(graph_features.mean()):
                print("❌ Graph tower produces NaN or Inf values!")
                return
            else:
                print("✅ Graph tower looks normal")
                
            # Test the mm_projector directly
            print("Testing mm_projector directly...")
            mm_projector_input = graph_features.unsqueeze(0)  # Add batch dimension
            print(f"MM projector input shape: {mm_projector_input.shape}")
            print(f"MM projector input mean: {mm_projector_input.mean()}, std: {mm_projector_input.std()}")
            
            # Apply mm_projector
            mm_projector_output = mm_projector_input
            for i, layer in enumerate(model.model.mm_projector):
                mm_projector_output = layer(mm_projector_output)
                print(f"MM projector layer {i} output mean: {mm_projector_output.mean()}, std: {mm_projector_output.std()}")
                if torch.isnan(mm_projector_output.mean()) or torch.isinf(mm_projector_output.mean()):
                    print(f"❌ MM projector layer {i} produces NaN or Inf values!")
                    return
                else:
                    print(f"✅ MM projector layer {i} looks normal")
            
            print("✅ MM projector looks normal")
            
        except Exception as e:
            print(f"❌ Graph tower test failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Process through multimodal pipeline
        result = model.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, [graph_data_obj]
        )
        
        print(f"Multimodal processing result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if result else 'None'}")
        
        if result and len(result) >= 5:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = result
            print(f"Processed input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"Inputs embeds shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
            
            # Check inputs_embeds for NaN values
            if inputs_embeds is not None:
                print(f"Inputs embeds mean: {inputs_embeds.mean()}, std: {inputs_embeds.std()}")
                if torch.isnan(inputs_embeds.mean()) or torch.isinf(inputs_embeds.mean()):
                    print("❌ Inputs embeds contain NaN or Inf values!")
                    return
                else:
                    print("✅ Inputs embeds look normal")
        else:
            print("❌ Multimodal processing failed")
            return
        
    except Exception as e:
        print(f"Error in multimodal processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass first
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            # Test the model's forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=input_ids  # Use input_ids as labels for testing
            )
            print(f"Forward pass successful! Output logits shape: {outputs.logits.shape}")
            print(f"Logits sample: {outputs.logits[0, 0, :10]}")
            
            # Check if logits contain meaningful values
            logits_mean = outputs.logits.mean()
            logits_std = outputs.logits.std()
            print(f"Logits mean: {logits_mean}, std: {logits_std}")
            
            if torch.isnan(logits_mean) or torch.isinf(logits_mean):
                print("❌ Logits contain NaN or Inf values!")
                
                # Check if the issue is with inputs_embeds
                print(f"Inputs embeds mean: {inputs_embeds.mean()}, std: {inputs_embeds.std()}")
                if torch.isnan(inputs_embeds.mean()) or torch.isinf(inputs_embeds.mean()):
                    print("❌ Inputs embeds contain NaN or Inf values!")
                else:
                    print("✅ Inputs embeds look normal")
                
                # Check if the issue is with the mm_projector
                print("Testing mm_projector...")
                try:
                    # Test the mm_projector directly
                    device = next(model.parameters()).device
                    dummy_input = torch.randn(1, 308, device=device, dtype=torch.float16)
                    print(f"Dummy input mean: {dummy_input.mean()}, std: {dummy_input.std()}")
                    
                    for i, layer in enumerate(model.model.mm_projector):
                        dummy_input = layer(dummy_input)
                        print(f"Layer {i} output mean: {dummy_input.mean()}, std: {dummy_input.std()}")
                        if torch.isnan(dummy_input.mean()) or torch.isinf(dummy_input.mean()):
                            print(f"❌ Layer {i} produces NaN or Inf values!")
                            break
                        else:
                            print(f"✅ Layer {i} looks normal")
                    
                except Exception as e:
                    print(f"❌ MM projector test failed: {e}")
                
            else:
                print("✅ Logits look normal")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate
    with torch.no_grad():
        try:
            if inputs_embeds is not None:
                # Use inputs_embeds instead of input_ids
                output_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=50,
                    use_cache=True
                )
            else:
                # Fallback to input_ids
                output_ids = model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=50,
                    use_cache=True
                )
            
            print(f"Generated output IDs shape: {output_ids.shape}")
            print(f"Generated output IDs: {output_ids}")
            
            # Decode
            if input_ids is not None:
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            else:
                # If using inputs_embeds, decode the entire output
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            outputs = outputs.strip()
            
            print(f"Generated response: '{outputs}'")
            
            if outputs == "" or outputs == "<unk>" or "unk" in outputs.lower():
                print("❌ Model is generating empty or <unk> responses!")
            else:
                print("✅ Model generated a proper response!")
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_gpm_generation()
