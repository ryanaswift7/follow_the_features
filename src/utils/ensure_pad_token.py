def ensure_pad_token(model, tokenizer):
    """
    Checks if a tokenizer has a pad token. If not, adds one, resizes the 
    model's embeddings, and syncs the configurations. Ensures left padding 
    for batched generation.

    Args:
        model: The instantiated Hugging Face model.
        tokenizer: The instantiated Hugging Face tokenizer.

    Returns:
        tuple: The updated (model, tokenizer).
    """
    # Enforce left padding (crucial for decoder-only models during generation)
    if tokenizer.padding_side != 'left':
        print(f"🔄 Changing padding side from '{tokenizer.padding_side}' to 'left'...")
        tokenizer.padding_side = 'left'

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