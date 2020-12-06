import torch


class ConstructBlockStrategy():
    def __init__(self):
        pass
    def construct(sample, *args, **kwargs):
        raise NotImplementedError


def get_masks_and_position_ids_blocklm(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
    loss_mask=None,
    attention_mask=None,
    transformer_xl=False,
    mem_length=None,
    construct_block_strategy=None,
):
    assert not transformer_xl and mem_length is None and attention_mask is None, 'blocklm does not support transformer-xl!'

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        loss_mask[data == eod_token] = 0.0
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i+1):, :(i+1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i+1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids