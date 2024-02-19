import torch
import torch.nn.functional as F
import pdb

TOPK=10 # topk for sparse tree (10 is a placeholder and it is sufficient)

def extract_last_valid_logits(logits: torch.Tensor, valid_length: torch.Tensor):
    """
    Extract logits of the last valid token for each sequence in the batch.

    Args:
        logits (torch.Tensor): Logits tensor of shape [batch_size, sequence_length, vocab_size], on GPU.
        valid_length (torch.Tensor): valid_length tensor of shape [batch_size], on GPU.

    Returns:
        torch.Tensor: Tensor containing the logits of the last valid token for each sequence, on GPU.
    """
    if logits.dim() == 3:
        batch_indices = torch.arange(logits.size(0), device=logits.device)  # Batch indices
        # Extract the logits of the last valid token for each sequence
        last_valid_logits = logits[batch_indices, valid_length - 1]
    elif logits.dim() == 4:
        batch_indices = torch.arange(logits.size(1), device=logits.device)  # Batch indices
        # Extract the logits of the last valid token for each sequence
        last_valid_logits = logits[:, batch_indices, valid_length - 1].transpose(1,0)        
    return last_valid_logits

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def initialize_medusa(input_ids, model, medusa_attn_mask, past_key_values, attention_mask=None):
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
        input_ids, attention_mask=attention_mask, past_key_values=past_key_values, output_orig=True, medusa_forward=True
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
    - None
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

def get_nucleus_one_token(logit, temperature, top_p):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def generate_candidates(medusa_logits, logits, tree_indices, retrieve_indices, temperature = 0, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = False, valid_length=None):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - medusa_logits (torch.Tensor): Logits from a specialized Medusa structure, aiding in candidate selection.
    - logits (torch.Tensor): Standard logits from a language model.
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - temperature (float, optional): Controls the diversity of the sampling process. Defaults to 0.
    - posterior_threshold (float, optional): Threshold for typical sampling. Defaults to 0.3.
    - posterior_alpha (float, optional): Scaling factor for the entropy-based threshold in typical sampling. Defaults to 0.09.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """

    if valid_length is not None:
        last_logits = extract_last_valid_logits(logits, valid_length)
        medusa_last_logits = extract_last_valid_logits(medusa_logits, valid_length)
    else:
        last_logits = logits
        medusa_last_logits = medusa_logits
    # Greedy decoding: Select the most probable candidate from the original logits.
    if temperature == 0 or fast:
        candidates_logit = torch.argmax(last_logits, dim=-1).unsqueeze(-1) ##logits: [bs,seq,vocab], candidates_logit:[bs,1]
    else:
        if sampling == 'typical':
            candidates_logit = get_typical_one_token(last_logits, temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus':
            candidates_logit = get_nucleus_one_token(last_logits, temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError
    
    # Extract the TOPK candidates from the medusa logits.
    candidates_medusa_logits = torch.topk(medusa_last_logits, TOPK, dim = -1).indices ##candidates_medusa_logits：[bs, medusa_head_num, TOPK]
    batch_size = candidates_logit.shape[0]
    # Combine the selected candidate from the original logits with the topk medusa logits.
    candidates = torch.cat([candidates_logit, candidates_medusa_logits.view(batch_size, -1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[:,tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((batch_size,1), dtype=torch.long, device=tree_candidates.device)], dim=-1)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[:,retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    # tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, tree_candidates

def update_position_id(medusa_position_ids, attention_mask, input_ids):
    bs = input_ids.shape[0]
    seqlen = medusa_position_ids.shape[0]
    medusa_position_ids_unsqueezed = torch.unsqueeze(medusa_position_ids, dim=0)
    medusa_position_ids_repeated = medusa_position_ids_unsqueezed.repeat(bs,1)
    valid_length = torch.unsqueeze(attention_mask.sum(dim=1), dim=-1)
    valid_length_repeated = valid_length.repeat(1, seqlen)
    position_ids = medusa_position_ids_repeated + valid_length_repeated
    return position_ids

def update_attention_mask(attention_mask, tree_candidates):
    bs = tree_candidates.shape[0]
    n = tree_candidates.shape[1]
    new_tokens = torch.ones((bs, n), dtype=attention_mask.dtype, device=attention_mask.device)
    extended_attention_mask = torch.cat((attention_mask, new_tokens), dim=1)
    return extended_attention_mask


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
    attention_mask
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """
    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    # position_ids = medusa_position_ids + input_ids.shape[1]
    position_ids = update_position_id(medusa_position_ids, attention_mask, input_ids)
    attention_mask = update_attention_mask(attention_mask, tree_candidates)
    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    tree_medusa_logits, outputs, tree_logits = model(
        tree_candidates,
        attention_mask=attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        medusa_forward=True,
    )
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[:, retrieve_indices]
    medusa_logits = tree_medusa_logits[:, :, retrieve_indices]
    return medusa_logits, logits, outputs

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask
     
def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = True
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
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
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
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:,:,:-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:,:,1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
            batch_size, num_path = candidates_accept_length.shape
            best_candidate = torch.zeros(batch_size, dtype=torch.long, device=candidates_accept_length.device)
            # Choose the best candidate based on the evaluated posterior probabilities
            accept_lengths = candidates_accept_length.max(dim=1)[0]
            for i in range(batch_size):
                if accept_lengths[i] != 0:
                    best_candidates = torch.where(candidates_accept_length[i] == accept_lengths[i])[0]
                    # Accept the best one according to likelihood
                    likelihood = torch.sum(
                        torch.log(candidates_prob[i, best_candidates, :accept_lengths[i]]), dim=-1
                    )
                    best_candidate[i] = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_lengths
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha, fast)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def generate_gather_mask(accept_length, max_accept_length):
    batch_size = accept_length.shape[0]
    range_tensor = torch.arange(max_accept_length, device='cuda:0').expand(batch_size, -1)
    gather_mask = (range_tensor < accept_length.unsqueeze(1))
    return gather_mask


def generate_gather_indices(gather_mask, max_accept_length, candidate_ids, prev_input_len):
    batch_size, _ = candidate_ids.shape
    output_indices = torch.full((batch_size, max_accept_length), -1, dtype=torch.long, device=candidate_ids.device)
    candidate_ids_ = candidate_ids[:, :max_accept_length] + prev_input_len
    output = torch.where(gather_mask, candidate_ids_, output_indices)
    return output


def select_new_tokens(candidates, best_candidate, gather_mask, max_accept_length, padding_id):
    batch_size, _, _ = candidates.shape
    candidates = candidates[:, :, :max_accept_length]
    best_paths = torch.gather(candidates, 1, best_candidate.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidates.size(2))).squeeze(1)
    default_ids = torch.full((batch_size, max_accept_length), padding_id, dtype=best_paths.dtype, device=best_paths.device)
    output = torch.where(gather_mask, best_paths, default_ids)
    return output


def gather_from_past_key_values(past_key_values_data, select_indices):
    layers, batch_size, head_num, _, hidden_size = past_key_values_data.shape
    seqlen = select_indices.shape[1]

    result_data = torch.zeros(layers, batch_size, head_num, seqlen, hidden_size, device=past_key_values_data.device, dtype=past_key_values_data.dtype)

    expanded_indices = select_indices.unsqueeze(0).unsqueeze(2).expand(layers, batch_size, head_num, seqlen)

    valid_indices_mask = expanded_indices != -1

    corrected_indices = torch.where(valid_indices_mask, expanded_indices, torch.zeros_like(expanded_indices))

    gathered_data = torch.gather(past_key_values_data, 3, corrected_indices.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_size))

    result_data = torch.where(valid_indices_mask.unsqueeze(-1), gathered_data, result_data)
    return result_data

## pad every step
def update_ids_new(input_ids, new_ids):
    input_ids = torch.cat([input_ids, new_ids], dim=-1)
    return input_ids

def update_mask(attention_mask, accept_length):
    range_tensor = torch.arange(accept_length.max().item(), device='cuda:0').expand(accept_length.shape[0], -1)
    new_attention_mask = (range_tensor < accept_length.unsqueeze(1)).to(int)
    attention_mask = torch.cat((attention_mask, new_attention_mask), dim=-1)
    return attention_mask

def update_kvcache(tgt, past_key_values_data, prev_input_len):
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    dst.copy_(tgt, non_blocking=True)

## avoid too much [PAD]
def update_ids_new(previous_ids, new_ids, padding_value=0):
    batch_size = previous_ids.shape[0]
    previous_seqlen = previous_ids.shape[1]
    new_seqlen = new_ids.shape[1]
    new_id_index = torch.arange(new_seqlen, device='cuda:0').expand(batch_size, -1)
    
    previous_mask = previous_ids != padding_value
    new_id_mask = new_ids != padding_value
   
    previous_valid_lengths = previous_mask.sum(dim=1)
    new_id_valid_lengths = new_id_mask.sum(dim=1)
    broad_max_output_len = previous_valid_lengths.max() + new_id_valid_lengths.max()
    tight_max_output_len = (previous_valid_lengths + new_id_valid_lengths).max()

    new_id_index = previous_valid_lengths.view(batch_size,-1) + new_id_index
    output = torch.full((batch_size, broad_max_output_len), padding_value, dtype=previous_ids.dtype, device=previous_ids.device)
    output[:, :previous_seqlen] = previous_ids
    output = output.scatter(1,new_id_index, new_ids)
    if tight_max_output_len < broad_max_output_len:
        output = output[:,:tight_max_output_len]
    return output, new_id_index

def update_mask_new(attention_mask, accept_length):
    batch_size = attention_mask.shape[0]
    previous_valid_lengths = attention_mask.sum(dim=1)
    new_valid_lengths = (previous_valid_lengths + accept_length).view(batch_size, -1)
    max_new_valid_length = new_valid_lengths.max()
    output = torch.arange(max_new_valid_length, dtype=attention_mask.dtype, device=attention_mask.device).expand(batch_size, -1)
    output = (output < new_valid_lengths).to(int)
    return output, max_new_valid_length

def update_kvcache_new(tgt, past_key_values_data, scatter_index):
    n_layers, _, num_head, _, hidden_size = tgt.shape
    expand_scatter_index = scatter_index.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(n_layers,-1,num_head,-1,hidden_size)
    past_key_values_data.scatter_(3, expand_scatter_index, tgt)

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
    attention_mask,
    padding_idx=0
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
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model. [layers, batch_size, head_num, max_seqlen, hidden_size]
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    accept_length += 1 ## accept_length > 0
    max_accept_length = accept_length.max().item()
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    candidate_ids = retrieve_indices[best_candidate]
    gather_mask = generate_gather_mask(accept_length, max_accept_length)
    select_indices = generate_gather_indices(gather_mask, max_accept_length, candidate_ids, prev_input_len)
    new_ids = select_new_tokens(candidates, best_candidate, gather_mask, max_accept_length, padding_id=padding_idx)
    # Append the tokens from the best candidate to the input sequence
    input_ids = update_ids_new(input_ids, new_ids)
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = gather_from_past_key_values(past_key_values_data, select_indices)
    # Destination tensor where the relevant past information will be stored
    # Copy relevant past information from the source to the destination
    update_kvcache(tgt, past_key_values_data, prev_input_len)
    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])    
    batch_indices = torch.arange(best_candidate.size(0), device=logits.device)
    if True:
        # Extract logits and medusa logits for the accepted tokens
        logits = logits[batch_indices, best_candidate, : max_accept_length]
        medusa_logits = medusa_logits[:, batch_indices, best_candidate, : max_accept_length]
        valid_length = accept_length
    else:
        # Extract logits and medusa logits for the last accepted tokens
        logits = logits[batch_indices, best_candidate, accept_length-1]
        medusa_logits = medusa_logits[:, batch_indices, best_candidate, accept_length-1]
        valid_length = None
    # Update the new token counter
    new_token += max_accept_length
    attention_mask = update_mask(attention_mask, accept_length)
    return input_ids, logits, medusa_logits, new_token, valid_length, attention_mask
