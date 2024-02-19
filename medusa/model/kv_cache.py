import torch
import copy

class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous lengths before adding new data
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        # 选取需要复制的数据
        tgt = self.data.index_select(dim, indices)
        prev_len = prev_length
        start_index = prev_len
        end_index = start_index + tgt.shape[dim]
        # 根据维度选取目标区域并复制数据
        if dim == 2:
            dst = self.data[:, :, :, start_index:end_index, :]
        elif dim == 3:
            dst = self.data[:, :, :, :, start_index:end_index]
        else:
            raise ValueError("Unsupported dimension for copying.")
        dst.copy_(tgt[:, :], non_blocking=True)   
        self.current_length.fill_(prev_length + tgt.shape[dim]) 

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data for batch_size > 1, and return the tensor
        truncated to the maximum current length across all batches.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated, assuming the first dimension is the batch size.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation and truncation to the maximum current length.
        """
        cur_len = copy.deepcopy(self.current_length)
        new_len = cur_len + tensor.size(dim)
        self.current_length.add_(tensor.shape[dim])
        if dim == 2:
            self.data[:, :, cur_len:new_len, :] = tensor[:,:,:,:]
            truncated_data = self.data[:, :, :self.current_length, :]
        elif dim == 3:
            self.data[:, :, :, cur_len:new_len] = tensor[:,:,:,:]
            truncated_data = self.data[:, :, :, :self.current_length]
        else:
            raise ValueError("Unsupported dimension for concatenation.")        
        return truncated_data


def initialize_past_key_values(model, batch_size=1):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing a tensor to store past keys and values for all layers
    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=model.device,
        dtype=model.dtype,
    )
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data
