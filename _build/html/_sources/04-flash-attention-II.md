# Flash Attention - II
This chapter assumes that you know about attention mechanism. If not please see this video, which provides a lot of info about how to model, train a GPT-2 from ground up, [Andrej Karpathy video](https://www.youtube.com/watch?v=l8pRSuU81PU).
This chapter compromises of:
1) Attention Pytorch Code.
2) Attention From Scratch Using Cuda.

## Attention Pytorch Code.

The below is the computation of Attention using pytorch.
```python
def forward(self, qkvr): // x is the qkv matrix
        B, T, three_C = qkvr.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = three_C // 3

        #### step 1: permute ####
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = split(qkvr) # split qkvr into q, k, v 
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        
        #### step 2: Dot product ########
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        ##### step 3: scale and mask ######
        # block_size is the max sequence length
        bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size) 
        att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        
        ##### step 4: Perform softmax operation ######
        att = F.softmax(att, dim=-1)
        
        #### Step 5: calculate y #######
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        #### step 6: unpermute final output ######
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return y
```

The cuda code we write will be made of these sub components, which is defined in the above code:
- Step 0: The int main() function where we define the host/device inputs, outputs and intermediate variables.
- step 1 - Permute: Splitting the input qkv into q, k, v. By calling the permute kernel.
- step 2 - Dot Product: Dot product of q and k to compute preattn.
- step 3 - Scale And Mask: Scale and mask the entries of preattn matrix.
- step 4 - Perform softmax operation.
- step 5 - Calculate y.
- step 6 - Unpermute y.

Lets start with step 0, where we define the host/device inputs, outputs and intermediate variables.

## Defining the inputs and outputs.
