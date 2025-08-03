import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference_stochastic, sim_matrix_inference, generate_embeds_per_video_id, sim_matrix_inference_stochastic_light_allops, generate_embeds_per_video_id_stochastic, np_softmax
from pympler import asizeof
import pickle

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 
        self.local_rank = config.local_rank
        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.save_eval = config.save_eval
        


    def _train_epoch(self, epoch):
        
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        

        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)

            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)
            data['audio'] = data['audio'].to(self.device)

            text_embeds, video_embeds_pooled, text_embeds_stochastic, text_mean, text_log_var, batch_text_features\
                = self.model(data, is_train=True)
            
            output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)

            
            with torch.no_grad():
                positive_sims = torch.diag(output)
                weights = positive_sims / positive_sims.sum()
          
            logit_scale = self.model.module.clip.logit_scale.detach()
            loss = self.loss(output, logit_scale, weights)

            video_embeds_pooled_avg = torch.mean(video_embeds_pooled,dim=1).squeeze()
            pointer = video_embeds_pooled_avg - text_embeds
            text_support = pointer / pointer.norm(dim=-1, keepdim=True) * torch.exp(text_log_var) + text_embeds
            output_support = sim_matrix_training(text_support, video_embeds_pooled, self.pooling_type)
            loss_support = self.loss(output_support, logit_scale, weights)

            loss_all = loss + loss_support * self.config.support_loss_weight
            loss_all.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.module.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            if self.config.noloss_record:
                pass
            else:
                if self.local_rank==0:
                    gen_log(model_path=self.config.model_path, log_name='log_tot_loss',
                            msg=loss_all.item())
                    gen_log(model_path=self.config.model_path, log_name='log_ori_loss',
                            msg=loss.item())
                    gen_log(model_path=self.config.model_path, log_name='log_sup_loss',
                            msg=loss_support.item())


            if batch_idx % self.log_step == 0:
                msg = ('Train Epoch: {} dl: {}/{} Total Loss: {:.6f}, Original Loss: {:.6f}, Support Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss_all.detach().item(),
                    loss.detach().item(),
                    loss_support.detach().item(),
                    ))
                if self.local_rank==0:
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)


            if batch_idx in eval_steps:

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

                else:
                    if self.local_rank == 0:
                        val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                    self.model.train()
                    if self.local_rank == 0:
                        if val_res['R1-window'] > self.best_window:
                            self.best_window = val_res['R1-window']

                        if val_res['R1'] > self.best:
                            self.best = val_res['R1']
                            self._save_checkpoint(epoch, save_best=True)

                        msg = (" Current Best Window Average R@1 is {}".format(self.best_window), " Current Best R@1 is {}\n\n".format(self.best))
 
                        gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

        res = {
            'loss_train':  total_loss / num_steps
        }


        return res

    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
            
        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)
                data['audio'] = data['audio'].to(self.device)
                text_embed, vid_embed, _, gate, _ = self.model.module(data, return_all_frames=True, is_train=False)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                all_vid_ids.extend(data['video_id'])

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                    
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            
            # Pool frames for inference once we have all texts and videos
            vid_embeds_pooled = self.model.module.pool_frames(text_embeds.to(self.device), vid_embeds.to(self.device))
            vid_embeds_pooled = vid_embeds_pooled.cpu()

            # build stochastic text embeds #########################################
            start_selection_time = time.time()

            # initialize text_embeds_stochastic_allpairs: to avoid data leakage, break vid-txt dependence by dataloader
            text_embeds_stochastic_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))  # [B_v, B_t, D]

            
            for (idx_vid, single_vid), single_vid_embed_pooled in tqdm(zip(enumerate(vid_embeds), vid_embeds_pooled), total=vid_embeds.shape[0]):
                single_vid_vec = single_vid.unsqueeze(0)
                single_vid_repeat = single_vid_vec.expand(text_embeds.shape[0], -1, -1)
                text_embeds_stochastic_allpairs[idx_vid, :, :], _, _, _ = self.model.module.stochastic(text_embeds.to(self.device), single_vid_repeat.to(self.device), training=False)
                torch.cuda.empty_cache()


            del text_embeds, vid_embeds
            gc.collect()

            end_selection_time = time.time()
            msg = (f'To compute all stochastic-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time} s\n')
            if self.local_rank==0:
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)
            
            
            # finish build stochastic text embeds #########################################
            
            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_stochastic(text_embeds_stochastic_allpairs,
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)
            
            del text_embeds_stochastic_allpairs
            del vid_embeds_pooled
            gc.collect()


            if self.config.save_memory_mode:
                start_sims = time.time()
                if self.local_rank==0:
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg='Use sim_matrix_inference_stochastic_light()')
                sims = sim_matrix_inference_stochastic_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.config.batch_size_split, self.config)
                end_sims = time.time()
                if self.local_rank==0:
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=f'batch size split = {self.config.batch_size_split}, sims compute time={end_sims-start_sims}')

            else:
                sims = sim_matrix_inference_stochastic(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)
          
            total_val_loss = total_val_loss / len(self.valid_data_loader)


            
            metrics = self.metrics
            res = metrics(sims)
            

            for m in res:
                self.window_metric[m].append(res[m])

            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            msg = (f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----",
                  f"R@1: {res['R1']} (window: {res['R1-window']})", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})",
                  )
            if self.local_rank==0:
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

            res['loss_val'] =  total_val_loss

            # add DSL
            if self.config.DSL:
                sims = sims * np_softmax(sims*100, axis=0)
                res_dsl = metrics(sims)

                msg = (f"-----DSL Val Epoch: {epoch}, dl: {step}/{num_steps}-----",
                  f"R@1: {res_dsl['R1']}", 
                  f"R@5: {res_dsl['R5']}", 
                  f"R@10: {res_dsl['R10']}",
                  f"MedR: {res_dsl['MedR']}",
                  f"MeanR: {res_dsl['MeanR']}",
                  )
                if self.local_rank==0:
                    gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

            return res
