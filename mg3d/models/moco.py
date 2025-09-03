import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, K, dim, device):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        # create the queue
        self.register_buffer("queue_1", torch.randn(K, dim).to(device))
        self.queue_1 = nn.functional.normalize(self.queue_1, dim=0)
        self.register_buffer("queue_1_ptr", torch.zeros(1, dtype=torch.long).to(device))

        self.register_buffer("queue_2", torch.randn(K, dim))
        self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)
        self.register_buffer("queue_2_ptr", torch.zeros(1, dtype=torch.long).to(device))
         
    @torch.no_grad()
    def enqueue_dequeue(self, keys_1, keys_2):
        # gather keys before updating queue
##        keys = concat_all_gather(keys)
 
        batch_size = keys_1.shape[0]
 
        ptr_1 = int(self.queue_1_ptr)
        ptr_2 = int(self.queue_2_ptr)
        assert self.K % batch_size == 0  # for simplicity
 
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_1[ptr_1:ptr_1 + batch_size, :] = keys_1
        ptr_1 = (ptr_1 + batch_size) % self.K  # move pointer
        self.queue_1_ptr[0] = ptr_1
        
        self.queue_2[ptr_2:ptr_2 + batch_size, :] = keys_2
        ptr_2 = (ptr_2 + batch_size) % self.K  # move pointer
        self.queue_2_ptr[0] = ptr_2        
        
        
class MoCo_SSM(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, K, dim1, dim2, device):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_SSM, self).__init__()
        
        self.K = K
        # create the queue
        self.register_buffer("queue_1", torch.randn(K, dim1).to(device))
        self.queue_1 = nn.functional.normalize(self.queue_1, dim=0)
        self.register_buffer("queue_1_ptr", torch.zeros(1, dtype=torch.long).to(device))

        self.register_buffer("queue_2", torch.randn(K, dim2))
        self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)
        self.register_buffer("queue_2_ptr", torch.zeros(1, dtype=torch.long).to(device))
         
    @torch.no_grad()
    def enqueue_dequeue(self, keys_1, keys_2):
        # gather keys before updating queue
##        keys = concat_all_gather(keys)
 
        batch_size = keys_1.shape[0]
 
        ptr_1 = int(self.queue_1_ptr)
        ptr_2 = int(self.queue_2_ptr)
##        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr_1 + batch_size) <= self.K:
            self.queue_1[ptr_1:ptr_1 + batch_size, :] = keys_1
            ptr_1 = (ptr_1 + batch_size) % self.K  # move pointer
            self.queue_1_ptr[0] = ptr_1
                
            self.queue_2[ptr_2:ptr_2 + batch_size, :] = keys_2
            ptr_2 = (ptr_2 + batch_size) % self.K  # move pointer
            self.queue_2_ptr[0] = ptr_2
        else:
            self.queue_1[ptr_1: :] = keys_1[:(self.K - ptr_1),:]
            self.queue_1[:(batch_size - self.K + ptr_1), :] = keys_1[(self.K - ptr_1):,:]
            ptr_1 = (ptr_1 + batch_size) % self.K  # move pointer
            self.queue_1_ptr[0] = ptr_1
            
            self.queue_2[ptr_2: :] = keys_2[:(self.K - ptr_2),:]
            self.queue_2[:(batch_size - self.K + ptr_2), :] = keys_2[(self.K - ptr_2):,:]               
            ptr_2 = (ptr_2 + batch_size) % self.K  # move pointer
            self.queue_2_ptr[0] = ptr_2
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output