import random
from random import sample

import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Optional


class BERTDataset(Dataset):    
    # @classmethod
    # def __init_gen_img_emb(cls, gen_img_path):
    #     if hasattr(cls, "__gen_img_path") and cls.__gen_img_path != gen_img_path:
    #         warnings.warn(f"Warning: Trying to reinitialize generated image embeddings with different path from before. \
    #             Ignoring incoming path \"{gen_img_path}\".")
    #     if not hasattr(cls, "GEN_IMG_EMB"):
    #         cls.GEN_IMG_EMB = torch.load(f"{gen_img_path}/gen_img_emb.pt") # dim : ((num_item)*512*3)
    #         cls.__gen_img_path = gen_img_path
    
    def __init__(self, user_seq, sim_matrix, num_user, num_item, num_cat,
                 gen_img_emb: torch.Tensor,
                 item_prod_type: torch.Tensor,
                 text_emb: Optional[torch.Tensor] = None,
                 neg_size: int = 50, 
                 neg_sample_size : int = 3, #negative sampling
                 max_len: int = 30, 
                 mask_prob: float = 0.15, 
                 num_gen_img: int = 1, 
                 img_noise: bool = False, 
                 std: float = 1, 
                 mean: float = 0,
                 mlp_cat: bool = False) -> None:
        self.user_seq = user_seq
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.sim_matrix = sim_matrix
        self.neg_size = neg_size                 #negative sampling
        self.neg_sample_size = neg_sample_size   #negative sampling
        self.num_gen_img = num_gen_img
        self.gen_img_emb = gen_img_emb
        self.item_prod_type = item_prod_type
        self.text_emb = text_emb
        self.img_noise = img_noise
        self.std = std
        self.mean = mean
        self.mlp_cat = mlp_cat
        self.num_cat = num_cat
        
    def sampler(self, item, user_seq):
        candidate = np.setdiff1d(self.sim_matrix[item][:3000], user_seq, assume_unique=True)
        return candidate[:self.neg_size+1] #negative sampling
    
    def get_img_emb(self, tokens, labels):
        item_ids = tokens.clone().detach()
        mask_index = torch.where(item_ids == self.num_item + 1)  # mask 찾기
        item_ids[mask_index] = labels[mask_index]  # mask의 본래 아이템 번호 찾기
        item_ids -= 1
        
        img_idx = sample([0, 1, 2], k=self.num_gen_img)  # 생성형 이미지 추출
        img_emb = torch.flatten(self.gen_img_emb[item_ids][:, img_idx, :], start_dim=-2, end_dim=-1)
        if self.img_noise:
            img_emb += torch.randn_like(img_emb) * self.std + self.mean
        elif self.mlp_cat:
            img_emb = self.item_prod_type[item_ids]
        elif self.text_emb is not None:
            if self.text_emb.shape[0] == self.num_item: # detail_text embedding
                img_emb = self.text_emb[item_ids]
            elif self.text_emb.shape[0] == self.num_cat:  # category embedding
                img_emb = self.text_emb[self.item_prod_type[item_ids]]
                
        # TODO: add idx_groups
        # if self.idx_groups is not None:
        #     f = lambda x: sample(self.idx_groups[x], k=1)[0] if x != -1 else -1
        #     item_ids = np.vectorize(f)(item_ids.detach().cpu())
        
                
        return img_emb

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = torch.tensor(self.user_seq[index], dtype=torch.long) + 1
        tokens = []
        labels = []
        negs = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:  # train
                prob /= self.mask_prob
                # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                if prob < 0.8:
                    tokens.append(self.num_item + 1)
                elif prob < 0.9:
                    tokens.append(random.choice(range(1, self.num_item + 1)))
                else:
                    tokens.append(s)
                labels.append(s)
                negs.append( self.sampler(s - 1, seq - 1) + 1)
            else:  # not train
                tokens.append(s)
                labels.append(0)
                negs.append(np.zeros(self.neg_sample_size))#3개 뽑기.

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        negs = negs[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        negs = np.concatenate([np.zeros((mask_len, self.neg_sample_size)), negs], axis=0)
        #3개 뽑기.
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        negs = torch.tensor(negs, dtype=torch.long)
        
        img_emb = self.get_img_emb(tokens, labels)
        

        return (
            tokens,
            img_emb,
            labels,
            negs,
        )


class BERTTestDataset(BERTDataset):
    def __init__(self, user_seq, sim_matrix, num_user, num_item, num_cat,
                 gen_img_emb: torch.Tensor,
                 item_prod_type: torch.Tensor,
                 text_emb: Optional[torch.Tensor] = None,
                 neg_size: int = 50, 
                 neg_sample_size : int = 3, #negative sampling
                 max_len: int = 30, 
                 num_gen_img: int = 1, 
                 img_noise: bool = False, 
                 std: float = 1, 
                 mean: float = 0,
                 mlp_cat: bool = False
                 ) -> None:
        super().__init__(
            user_seq=user_seq,
            sim_matrix=sim_matrix,
            num_user=num_user,
            num_item=num_item,
            num_cat=num_cat,
            gen_img_emb=gen_img_emb,
            item_prod_type=item_prod_type,
            text_emb=text_emb,
            neg_size=neg_size,
            neg_sample_size=neg_sample_size,
            max_len=max_len,
            num_gen_img=num_gen_img,
            img_noise=img_noise,
            std=std,
            mean=mean,
            mlp_cat=mlp_cat
        )

    def __getitem__(self, index):
        tokens = torch.tensor(self.user_seq[index], dtype=torch.long) + 1

        labels = [0 for _ in range(self.max_len)]
        negs = np.zeros((self.max_len, self.neg_sample_size)) #3개 뽑기
        labels[-1] = tokens[-1].item()  # target
        negs[-1] = self.sampler(labels[-1] - 1, tokens - 1) + 1 #3개 뽑기
        tokens[-1] = self.num_item + 1  # masking

        tokens = tokens[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # padding
        tokens = torch.concat((torch.zeros(mask_len, dtype=torch.long), tokens), dim=0)
        
        labels = torch.tensor(labels, dtype=torch.long)
        negs = torch.tensor(negs, dtype=torch.long)
        
        img_emb = self.get_img_emb(tokens, labels)

        return index, tokens, img_emb, labels, negs
