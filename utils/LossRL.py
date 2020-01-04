import torch
import numpy as np
from utils import cuda

# import nltk


from cotk.metric.bleu import _sentence_bleu # Maybe use another bleu metric ?



class PolicyGradientLoss(torch.nn.modules.loss._Loss):
    """
        From the paper https://arxiv.org/pdf/1805.09461v4.pdf page 9, Eq. (12-18)
    """
    def __init__(self, metric = None, baseline_reward='mean'):
        super(PolicyGradientLoss, self).__init__()

        if metric is None:
            self.metric = _sentence_bleu
        else:
            self.metric = metric
        self.baseline_reward   = baseline_reward


    def forward(self, generated_sentence, reference_sentence, generated_distribution, sentence_length):
        """
        
            generated_sentence     -> keys of the tokens generated
            reference_sentence     -> keys of the target tokens
            generated_distribution -> input of the softmax (projection over the vocabulary)
            sentence_length        -> length of the sentences (/!\ type = np.ndarray)
            
            ---------------------------------------------------------------------------------
            generated_sentence     = nb_sample x max_sent_length     x batch_size
            reference_sentence     =             max_ref_sent_length x batch_size
            generated_distribution = nb_sample x max_sent_length     x batch_size x vocab_size
            sentence_length        = nb_sample                        x batch_size
            ---------------------------------------------------------------------------------

        """
        # If   generated_sentence =     generated_sent_length x batch_size
        # Then generated_sentence = 1 x generated_sent_length x batch_size
        if generated_sentence.dim == 2: 
            generated_sentence     = generated_sentence.unsqueeze(0)
            generated_distribution = generated_distribution.unsqueeze(0)
            sentence_length        = sentence_length.unsqueeze(0)
               

        
        # We dont keep the value generated that have a length > max_ref_sent_length, to avoid usless computation
        # Rem: I am note sure whether it should be used or not. It is not necessary
        # generated_sentence     = generated_sentence[:,     0:generated_sentence.shape[1], :]
        # generated_distribution = generated_distribution[:, 0:generated_sentence.shape[1], :, :]


        self.nb_sample       = nb_sample       = generated_sentence.shape[0] # = N
        self.max_sent_length = max_sent_length = generated_sentence.shape[1] # = T
        self.batch_size      = batch_size      = generated_sentence.shape[2] # = B

        self.generated_sentence     = generated_sentence     # N x T x B
        self.reference_sentence     = reference_sentence     #     T x B
        self.generated_distribution = generated_distribution # N x T x B x C
        self.sentence_length        = sentence_length        # N     x B
        
        
        
        # (N x T x B x C | N x T x B) -> N x T x B
        logPi = self._logPi(generated_distribution, generated_sentence)

        # N x T x B -> N x T x B
        masked_logPi = self._mask_logPi(logPi, sentence_length, nb_sample, max_sent_length, batch_size)

        # N x T x B -> N x B
        sumLogPi = torch.einsum('ntb->nb', masked_logPi)

        # (N x T x B | N x T x B) -> N x T x B
        reward = self._reward(generated_sentence, reference_sentence)

        # N x B -> 1 (computes mean over the samples and over the batch)
        loss = torch.mean(sumLogPi * reward)

        return loss

        
        
        
    def _logPi(self, generated_distribution, generated_sentence):
        """
            inputs shape : N x T x B
        """
        # N x T x B( x C) -> B (x C) x N x T
        permutation_distribution = generated_distribution.permute(2, 3, 0, 1)
        permutation_sentence     = generated_sentence.permute(    2,    0, 1)

        # B (x C) x N x T -> B x N x T
        logPi = -torch.nn.functional.cross_entropy(permutation_distribution, permutation_sentence, reduction='none') 
        # TODO: check if this is right

        # B x N x T -> N x T x B
        return logPi.permute(1, 2, 0)
    
    
    
    def _mask_logPi(self, logPi, sentence_length, nb_sample, max_sent_length, batch_size):
        """
            logPi shape : N x T x B
        """

        # T
        arange = np.arange(max_sent_length)

        # T x N * B
        arange_repeat = arange.repeat(nb_sample*batch_size)

        # T x N x B
        arange_repeat = cuda(torch.tensor(arange_repeat.reshape([max_sent_length, nb_sample, batch_size]))).long()

        # T x N x B
        mask = (arange_repeat >= sentence_length.unsqueeze(0))

        # N x T x B
        mask = mask.permute(1,0,2) # To have the same shape as logPi

        # value masked (=0) when i > length_sentence, i.e. when the sentence stops
        # N x T x B
        return logPi.masked_fill(mask, float('0'))


        
    def _reward(self, generated_sentence, reference_sentence):
        """
            inputs shape : (N x T x B |  T x B)
        """
        # (N x T x B | N x T x B) -> N x T x B
        r = self._batch_metric(generated_sentence, reference_sentence)

        # 1 x B
        if self.baseline_reward == "mean":
            rb = torch.mean(r, dim=0, keepdim=True)-0.01 
        else: #TODO implement other baseline_reward, this "else" is just an example (from paper page 8)
            baseline_value = 0.2
            rb = cuda(torch.ones(r.shape).float()*baseline_value)

        # N x B
        return r-rb
        

    
    def _batch_metric(self, generated_sentence, reference_sentence):
        """
            Returns the metric applied on every sentence (over the samples and the batch)
            inputs shape : (N x T x B |  T x B)
            output shape :  N x B
        """
        # T x B -> N x T x B
        reference_sentence = reference_sentence.unsqueeze(0).expand(self.nb_sample, -1, -1)

        # N x T x B -> N*B x T
        perm_generated = generated_sentence.permute(0,2,1).reshape([-1, generated_sentence.shape[1]])
        perm_reference = reference_sentence.permute(0,2,1).reshape([-1, reference_sentence.shape[1]])
        
        list_bleu = []
        for hypothesis, reference in zip(perm_generated, perm_reference): # for N*B elements
            # [T], T -> 1
            list_bleu.append(self.metric(([reference], hypothesis))) 
            
        # N*B
        tensor_bleu = cuda(torch.tensor(list_bleu).float())
        
        # N*B -> N x B
        return tensor_bleu.reshape([self.nb_sample, self.batch_size])       
        








# # TODO : Should be stored somewhere else
# def oneHot(tensor_id, vocab_size):
#     return cuda(torch.eye(vocab_size)[tensor_id])


# class BlueMetricVIACE(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, generated_sentence, reference_sentence, generated_distribution_CE):
#         # generated_sentence and reference_sentence are used for the metric
#         # generated_distribution_CE has to be added as a parameter for the backward step
#         reference  = [list(reference_sentence.detach().cpu().numpy())]
#         hypothesis =  list(generated_sentence.detach().cpu().numpy())
#         value = torch.tensor([nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)])
#         ctx.save_for_backward(value)
#         return value

#     @staticmethod
#     def backward(ctx, grad_output):
#         value, = ctx.saved_tensors
#         rb = torch.tensor([0.7]) # TODO : set a better rb
#         return None, None, value-rb
#         # No grad for the sentences, only for the distribution
    
    
# class BlueMetric(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, generated_sentence, reference_sentence, generated_distribution):
#         # generated_sentence and reference_sentence are used for the metric
#         # generated_distribution has to be added as a parameter for the backward step
        
#         reference  = [list(reference_sentence.detach().cpu().numpy())]
#         hypothesis =  list(generated_sentence.detach().cpu().numpy())
#         value = cuda(torch.tensor([nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)]).float())
        
#         ctx.save_for_backward(generated_distribution, generated_sentence, value)
#         return value
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         generated_distribution, generated_sentence, value, = ctx.saved_tensors
#         rb = cuda(torch.tensor([0.2])) # TODO : set a better rb
#         temp = torch.nn.functional.softmax(generated_distribution, -1) - oneHot(generated_sentence, generated_distribution.shape[-1])
#         return None, None, temp*(value-rb)
#         # No grad for the sentences, only for the distribution
    
    




# class PolicyGradientLoss(torch.nn.modules.loss._Loss):
#     """
#         From the paper https://arxiv.org/pdf/1805.09461v4.pdf page 8
#     """
#     def __init__(self, metric, via_cross_entropy = False, size_average=None, reduce=None):
#         super(PolicyGradientLoss, self).__init__(size_average, reduce)
#         self.metric = metric
#         self.via_cross_entropy = via_cross_entropy
        
#     def forward(self, generated_sentence, reference_sentence, generated_distribution):
#         """
#             The forward is the metric value between the input and the target

#             -------------------------------------------------------------
#             generated_sentence     = batch_size x sentence_length
#             generated_distribution = batch_size x sentence_length x vocab_size
#             reference_sentence     = batch_size x sentence_length
#             -------------------------------------------------------------
#         """
#         self.generated_sentence     = generated_sentence
#         self.generated_distribution = generated_distribution
#         self.reference_sentence     = reference_sentence
        
        
#         if self.via_cross_entropy:# Same result but harder to understand what happens and seems to be ~20 slower
#             generated_distribution_CE = torch.nn.CrossEntropyLoss()(self.generated_distribution, self.generated_sentence) 
#             self.metric = BlueMetricVIACE.apply
#             self.lossMetric = self._evaluate(generated_sentence, reference_sentence, generated_distribution_CE)
#         else:
#             self.metric = BlueMetric.apply
#             self.lossMetric = self._evaluate(generated_sentence, reference_sentence, self.generated_distribution)
#         return self.lossMetric
        
#     def _evaluate(self, generated_sentence, reference_sentence, generated_distribution, batch = True):
#         if batch:
#             return torch.mean(torch.cat([self.metric(ref, hyp, distrib) for ref, hyp, distrib in zip(generated_sentence, 
#                                                                                                      reference_sentence, 
#                                                                                                      generated_distribution)]).float())
#         return self.metric(generated_sentence, reference_sentence, generated_distribution)



