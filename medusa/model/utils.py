import torch


def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers related to the Medusa structure.
    Split each part for readability.

    Explanation of each buffer in the returned dictionary:
    1. tree_indices: Represents indices that map items from a linear list to a tree structure.
    2. medusa_attn_mask: The attention mask designed specifically for the Medusa structure, ensuring proper attention computation.
    3. medusa_position_ids: Denotes the position identifiers used within the Medusa structure.
    4. retrieve_indices: Provides indices that map items from a tree structure back to their original positions in a cartesian product.
    5. list_indices: Represents indices mapping items from a tree back to a list. This is intended for a future feature and is currently under testing.

    Args:
        medusa_choices (torch.Tensor): A tensor containing choices for the Medusa structure.
        device (str, optional): Target device for the generated buffers. Defaults to "cuda".

    Returns:
        dict: A dictionary containing several buffer tensors for the Medusa structure.
    """
    medusa_choices = torch.tensor(medusa_choices)
    cumulative_product = torch.cumprod(medusa_choices, dim=0)
    cumulative_sum = torch.cumsum(medusa_choices, dim=0)
    medusa_len = cumulative_product.sum().item()
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)

    # 1. Generate tree indices based on the Medusa choices
    medusa_indices = torch.arange(cumulative_sum[-1])
    tree_indices = []
    prev_cumsum = 0
    prev_cumprod = 1
    for i in range(medusa_choices.size(0)):
        cumsum = cumulative_sum[i].item()
        cumprod = cumulative_product[i].item()
        slice = medusa_indices[prev_cumsum:cumsum].repeat(prev_cumprod, 1).flatten()
        tree_indices += slice.tolist()
        prev_cumsum = cumsum
        prev_cumprod = cumprod

    # 2. Update the Medusa attention mask
    prev_cumprod_sum = -1
    for i in range(medusa_choices.size(0)):
        cumprod_sum = cumulative_product[:i].sum().item()
        if prev_cumprod_sum != -1:
            parent_idx = (
                torch.arange(prev_cumprod_sum, cumprod_sum)
                .repeat(medusa_choices[i], 1)
                .transpose(0, 1)
                .flatten()
            )
            medusa_attn_mask[
                cumprod_sum : cumprod_sum + parent_idx.size(0)
            ] += medusa_attn_mask[parent_idx]
        prev_cumprod_sum = cumulative_product[:i].sum().item()

    # 3. Generate Medusa position IDs
    medusa_position_ids = []
    for i in range(medusa_choices.size(0)):
        medusa_position_ids += [i] * cumulative_product[i]

    # 4. Generate retrieval indices for Medusa structure verification
    medusa_len_prod = torch.prod(medusa_choices).item()
    retrieve_indices = torch.zeros(
        medusa_len_prod, len(medusa_choices), dtype=torch.long
    )
    prev_cumprod_sum = 0
    for i in range(medusa_choices.size(0)):
        cumprod_sum = cumulative_product[: i + 1].sum().item()
        retrieve_indices[:, i] = (
            torch.arange(prev_cumprod_sum, cumprod_sum)
            .repeat(medusa_len_prod // (cumprod_sum - prev_cumprod_sum), 1)
            .transpose(0, 1)
            .flatten()
        )
        prev_cumprod_sum = cumprod_sum

    # 5. Generate list indices for Medusa structure
    list_indices = []
    cumulative_product = torch.cumprod(medusa_choices, dim=0)
    cumulative_product_max = torch.max(cumulative_product)
    prev_cumprod_sum = 0

    for i in range(medusa_choices.size(0)):
        current_indices = torch.arange(
            prev_cumprod_sum, prev_cumprod_sum + medusa_choices[i]
        )
        current_indices = current_indices.repeat(
            cumulative_product[i] // medusa_choices[i], 1
        ) + torch.arange(cumulative_product[i] // medusa_choices[i]).unsqueeze(
            -1
        ) * current_indices.size(
            0
        )
        current_indices = current_indices.repeat(
            cumulative_product_max // (cumulative_product[i] // medusa_choices[i]), 1
        )
        list_indices.append(current_indices)
        prev_cumprod_sum += cumulative_product[i]
    list_indices = torch.cat(list_indices, dim=1).transpose(0, 1)

    # Compile all the buffers into a dictionary
    ret = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        "list_indices": list_indices,
    }

    # Convert all items in the dictionary to tensors and move them to the specified device
    ret = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in ret.items()
    }
    return ret


def initialize_medusa(input_ids, model, medusa_attn_mask, past_key_values):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    medusa_logits, outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    model.base_model.model.medusa_mask = medusa_attn_mask
    return medusa_logits, logits


def reset_medusa_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - past_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(medusa_logits, logits, medusa_topk, tree_indices, temperature):
    """
    Generates candidate tokens based on the Medusa logits and original logits.

    This function performs a greedy decoding on the original logits to retrieve
    the most likely token. For the Medusa logits, it retrieves the top-k tokens
    as specified by the `medusa_topk` argument. Finally, the function reshapes
    and matches these candidates based on the tree structure defined by `tree_indices`.

    Args:
    - medusa_logits (torch.Tensor): Output tensor of shape (medusa, batch_size, vocabulary_size)
      representing the logits from Medusa layers.
    - logits (torch.Tensor): Original logits tensor of shape (batch_size, sequence_length, vocabulary_size).
    - medusa_topk (list of int): Contains the number of top-k tokens to consider for each Medusa layer.
    - tree_indices (list or torch.Tensor): Index mapping from a flattened list to tree structure.
    - temperature (float): Scaling factor to modulate the logits' values before generating candidates (not used in this function but kept for future extensions).

    Returns:
    - candidates (torch.Tensor): Cartesian product of candidate tokens across Medusa layers.
    - tree_candidates (torch.Tensor): Reshaped candidates matched to the tree structure.
    """
    # Greedy decoding for original logits
    candidates = [torch.argmax(logits[:, -1]).unsqueeze(0)]
    for i in range(medusa_logits.shape[0]):
        candidate_i = torch.topk(medusa_logits[i, 0, -1], medusa_topk[i]).indices
        candidates.append(candidate_i)
    candidates_flat = torch.cat(candidates)
    candidates = torch.cartesian_prod(*candidates)
    tree_candidates = candidates_flat[tree_indices].unsqueeze(0)
    return candidates, tree_candidates


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decodes the token sequences using a tree-based approach with Medusa layers.

    Given the candidates for token sequences and the current past key values, the function
    decodes the sequences using the model's Medusa layers and retrieves the logits
    corresponding to the desired positions in the sequence.

    Args:
    - model (nn.Module): The main model with Medusa layers.
    - tree_candidates (torch.Tensor): Candidate tokens for the current decoding step based on the tree structure.
    - past_key_values (list of torch.Tensor): List of past key-value states to use for autoregressive decoding.
    - medusa_position_ids (list or torch.Tensor): Position IDs for the Medusa structure.
    - input_ids (torch.Tensor): The input token sequences of shape (batch_size, sequence_length).
    - retrieve_indices (list or torch.Tensor): Indices mapping from tree to cartesian product, used to reorder the logits.

    Returns:
    - medusa_logits (torch.Tensor): Medusa logits corresponding to the current decoding step.
    - logits (torch.Tensor): Original logits for the current step.
    - outputs (tuple): Intermediate model outputs.
    """

    # Compute new position IDs based on the Medusa structure and current input sequence length
    position_ids = medusa_position_ids + input_ids.shape[1]
    # Decode the tree candidates using the model
    tree_medusa_logits, outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    # Reorder the logits based on the retrieve_indices for consistency
    logits = tree_logits[0, retrieve_indices]
    medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
    return medusa_logits, logits, outputs


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold, posterior_alpha
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    # Calculate posterior probabilities and thresholds for candidate selection
    posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
    candidates_prob = torch.gather(
        posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    posterior_entropy = -torch.sum(
        posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
    )  # torch.sum(torch.log(*)) is faster than torch.prod
    threshold = torch.minimum(
        torch.ones_like(posterior_entropy) * posterior_threshold,
        torch.exp(-posterior_entropy) * posterior_alpha,
    )
    posterior_mask = candidates_prob > threshold
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

    # Choose the best candidate based on the evaluated posterior probabilities
    accept_length = candidates_accept_length.max()
    if accept_length == 0:
        # If no candidates are accepted, just choose the first one
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidates = torch.where(candidates_accept_length == accept_length)[0]
        # Accept the best one according to likelihood
        likelihood = torch.sum(
            torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
        )
        best_candidate = best_candidates[torch.argmax(likelihood)]
    return best_candidate, accept_length


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_length : accept_length + 1
    ]
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, medusa_logits, new_token
