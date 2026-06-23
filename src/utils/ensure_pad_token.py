def ensure_pad_token(model, tokenizer, is_inference: bool):
    """
    Checks if a tokenizer has a pad token. If not, adds one, resizes the 
    model's embeddings, and syncs the configurations. Sets the correct 
    padding side based on whether the model is used for training or inference.

    Args:
        model: The instantiated Hugging Face model.
        tokenizer: The instantiated Hugging Face tokenizer.
        is_inference (bool): True if batched generation, False if training.

    Returns:
        tuple: The updated (model, tokenizer).
    """
    # Enforce left padding for inference, right padding for training
    target_padding_side = 'left' if is_inference else 'right'
    
    if tokenizer.padding_side != target_padding_side:
        print(f"🔄 Changing padding side from '{tokenizer.padding_side}' to '{target_padding_side}'...")
        tokenizer.padding_side = target_padding_side

    new_pad_token = '[PAD]'
    if tokenizer.pad_token_id is None:
        print(f"⚠️ Pad token missing. Adding '{new_pad_token}' and resizing embeddings...")
        
        # 1. Add the special token to the tokenizer
        tokenizer.add_special_tokens({'pad_token': new_pad_token})
        
        # 2. Resize the model embeddings to prevent out-of-bounds errors
        model.resize_token_embeddings(len(tokenizer))
        
        # 3. Align the model configurations with the new tokenizer value
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 4. Update generation config if it exists
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            
        print(f"✅ Pad token added. New ID: {tokenizer.pad_token_id}")
    else:
        print(f"✅ Pad token already exists. ID: {tokenizer.pad_token_id}. No token changes made.")
        
    return model, tokenizer


def ensure_pad_token_hooked_transformer(model, is_inference: bool):
    """
    Ensures a HookedTransformer's tokenizer has a pad token by defaulting 
    to the EOS token if missing. Sets correct padding side.
    
    Args:
        model: Instantiated HookedTransformer.
        is_inference (bool): True for batched generation, False for training.
    """
    target_padding_side = 'left' if is_inference else 'right'
    
    if model.tokenizer.padding_side != target_padding_side:
        print(f"🔄 Changing padding side from '{model.tokenizer.padding_side}' to '{target_padding_side}'...")
        model.tokenizer.padding_side = target_padding_side

    if model.tokenizer.pad_token_id is None:
        print("⚠️ Pad token missing. Setting pad_token to eos_token...")
        
        # Reuse EOS token as PAD token (Standard practice for HookedTransformers)
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        
        print(f"✅ Pad token set to EOS. ID: {model.tokenizer.pad_token_id}")
    else:
        print(f"✅ Pad token already exists. ID: {model.tokenizer.pad_token_id}")

    return model, model.tokenizer