import os
import json
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import torch

from tqdm import tqdm
from pprint import pprint

from utils.tool import generate_Gen_mask, mean_the_denoising_process
from utils.metrics import *


def test_specific(config, pre_model, diff_model, test_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Evaluation. :")
        eval_metrics = {}
        # Resume: load already-done noise levels to skip them
        if os.path.exists(logger.exp_path):
            try:
                with open(logger.exp_path, "r") as f:
                    _existing = json.load(f)
                for _k, _v in _existing.get("eval_metrics_specific", {}).items():
                    eval_metrics[float(_k)] = _v
            except Exception:
                pass
        for noiser_factor in config.get("noise_list", [0, 0.25, 0.5, 0.75, 1]):
            if noiser_factor in eval_metrics:
                print(f"  Skipping noise={noiser_factor} (already done)")
                continue
            gen_dir = logger.generation_directory + "UK_noise" + str(noiser_factor) + "/"
            os.makedirs(gen_dir, exist_ok=True)

            # Mid-noise checkpoint: resume from last completed city
            ckpt_dir = os.path.join("exp", "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            exp_name = config["exp_name"]
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_n{noiser_factor}_partial.pkl")
            Gen_metrics = []
            start_idx = 0
            if os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, "rb") as f:
                        ckpt = pickle.load(f)
                    Gen_metrics = ckpt["Gen_metrics"]
                    start_idx = ckpt["next_idx"]
                    print(f"  Resuming noise={noiser_factor} from city {start_idx}/{len(test_Dloader)} ({len(Gen_metrics)} cities done)")
                except Exception:
                    Gen_metrics = []
                    start_idx = 0

            for i, batch in tqdm(enumerate(test_Dloader), total=len(test_Dloader), desc="noise-"+str(noiser_factor)):
                if i < start_idx:
                    continue
                geoid, batched_graph, batched_dis, ods = batch
                # n
                n = batched_graph.ndata["nfeat"].to(config["device"])
                # e
                dim = sum([x.shape[0] for x in ods])
                batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"])
                l, r = 0, 0
                for j, od in enumerate(ods):
                    l = r
                    r = r + od.shape[0]
                    batched_od[l:r, l:r] = ods[j]
                    batchlization[l:r, l:r] = 1
                e = batched_od.to(config["device"])
                # img introduce
                n_attr = n[:, 0: config["n_indim"]]
                noisy_attr = n_attr * noiser_factor + torch.randn_like(n_attr) * (1 - noiser_factor)
                n_img = n
                with torch.no_grad():
                    pred_attr = pre_model(n_img, noisy_attr)
                if config["if-ImgAttrAug"] == 0:
                    pred_attr = pred_attr * 0
                n = torch.cat([n_img, noisy_attr, pred_attr], dim=1)
                net = (n, e)
                batched_dis = batched_dis.to(config["device"])

                diff_model.eval()
                mask_n, mask_e = generate_Gen_mask(n, e)
                mask_n, mask_e = mask_n.to(config["device"]), mask_e.to(config["device"])
                masks = (mask_n, mask_e)
                c = (net, masks, batched_dis, batchlization)
                e_hats = []
                for _ in range(config["sample_times"]):
                    denoising_seq = diff_model.DDIM_sample_loop(n.shape, e.shape, c)
                    net_hat = denoising_seq[-1]
                    _, e_hat = net_hat
                    e_hats.append(e_hat.cpu().numpy())
                e_hat = np.mean(np.stack(e_hats), axis=0)
                od_hat = e_hat
                od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
                od_hat = scalers["od_normer"].inverse_transform(od_hat)
                for j in range(od_hat.shape[0]):
                    od_hat[j, j] = 0
                od_hat[od_hat < 0] = 0
                od_hat = np.floor(od_hat)
                np.save(gen_dir + geoid[0] + ".npy", od_hat)
                Gen_metrics.extend(eval_gen(config, logger, geoid, ods, diff_model, net, batched_dis, batchlization, scalers))

                # Save mid-noise checkpoint after each city
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"Gen_metrics": Gen_metrics, "next_idx": i + 1}, f)

            eval_metrics_p = average_listed_metrics(Gen_metrics)
            eval_metrics[noiser_factor] = eval_metrics_p
            # save after each noise level so partial results survive walltime kills
            if os.path.exists(logger.exp_path):
                with open(logger.exp_path, "r") as f:
                    log = json.load(f)
            else:
                log = {"train_log": {}}
            log["eval_metrics_specific"] = eval_metrics
            with open(logger.exp_path, "w") as f:
                json.dump(log, f, indent=4)
            # Remove mid-noise checkpoint (noise complete)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"  Noise={noiser_factor} complete, checkpoint removed.")

        pprint(eval_metrics)
    print("--------------------------------------------------")


def test(config, pre_model, diff_model, test_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Evaluation. :")
        eval_metrics = {}
        for noiser_factor in config.get("noise_list", [0, 0.25, 0.5, 0.75, 1]):
            logger.noiser_factor = noiser_factor
            Gen_metrics = []
            for i, batch in tqdm(enumerate(test_Dloader), total=len(test_Dloader), desc="noise-"+str(noiser_factor)):
                geoids, batched_graph, batched_dis, ods = batch
                # n
                n = batched_graph.ndata["nfeat"].to(config["device"])
                # e
                dim = sum([x.shape[0] for x in ods])
                batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"]) # There are zero-noise from blocking the OD matrix.
                l, r = 0, 0
                for i, od in enumerate(ods):
                    l = r
                    r = r + od.shape[0]
                    batched_od[l:r, l:r] = ods[i]
                    batchlization[l:r, l:r] = 1
                e = batched_od.to(config["device"])
                # img introduce
                n_attr = n[:, 0: config["n_indim"]]
                noisy_attr = n_attr * noiser_factor + torch.randn_like(n_attr) * (1 - noiser_factor)
                n_img = n[:, config["n_indim"]:]
                with torch.no_grad():
                    pred_attr = pre_model(n_img, noisy_attr)
                if config["if-ImgAttrAug"] == 0:
                    pred_attr = pred_attr *0
                n = torch.cat([n_img, noisy_attr, pred_attr], dim=1)
                # net
                net = (n, e)
                # dis
                batched_dis = batched_dis.to(config["device"])

                # metrics of one batch
                Gen_metrics.extend(eval_gen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers))
            
            # summary metrics and save log
            eval_metrics_p = average_listed_metrics(Gen_metrics)
            eval_metrics[noiser_factor] = eval_metrics_p
        pprint(eval_metrics)
        with open(logger.exp_path, "r") as f:
            log = json.load(f)
        log["eval_metrics"] = eval_metrics
        with open(logger.exp_path, "w") as f:
            json.dump(log, f, indent=4)

    print("--------------------------------------------------")
    
def test_one_city(config, city, scalers, diff_model):
    # load data
    nfeat = np.load(config["data_path"] + city + "/nfeat/nfeat.npy")
    dis = np.load(config["data_path"] + city + "/adj/dis.npy")
    od = np.load(config["data_path"] + city + "/od/od.npy")

    # normalization
    nfeat = scalers["feat"].transform(nfeat)
    dis = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    # Tensor
    nfeat = torch.from_numpy(nfeat).float().to(config["device"])
    dis = torch.from_numpy(dis).float().to(config["device"])

    # generation

def eval_gen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers):
    diff_model.eval()
    n, e = net
    mask_n, mask_e = generate_Gen_mask(n, e)
    mask_n, mask_e = mask_n.to(config["device"]), mask_e.to(config["device"])
    masks = (mask_n, mask_e)
    c = (net, masks, batched_dis, batchlization)

    # generate
    n_hats, e_hats = [], []
    denoising_seqs = []
    for _ in range(config["sample_times"]):
        denoising_seq = diff_model.DDIM_sample_loop(n.shape, e.shape, c)
        denoising_seqs.append(denoising_seq)
        net_hat = denoising_seq[-1]
        n_hat, e_hat = net_hat
        n_hats.append(n_hat)
        e_hats.append(e_hat)
    denoising_seq = mean_the_denoising_process(denoising_seqs)
    n_hat = torch.mean(torch.stack(n_hats), dim=0)
    e_hat = torch.mean(torch.stack(e_hats), dim=0)
    n_hat, e_hat = n_hat.cpu().numpy(), e_hat.cpu().numpy()


    # eval
    mask_n, mask_e = mask_n.cpu().numpy(), mask_e.cpu().numpy()
    all_metrics = []
    l, r = 0, 0
    for idx, od in enumerate(ods):
        l = r
        r = r + od.shape[0]
        # extract
        od_hat = e_hat[l:r, l:r]
        tmp_mask_e = mask_e[l:r, l:r]

        if np.sum(tmp_mask_e == 0) == 0:
            continue
        
        od = scalers["od"].inverse_transform(od.cpu().numpy().reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        od = scalers["od_normer"].inverse_transform(od)
        od[od < 0] = 0
        od_hat = scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape([od_hat.shape[0], od_hat.shape[1]])
        od_hat = scalers["od_normer"].inverse_transform(od_hat)
        for i in range(od_hat.shape[0]): # consistant with groundtruth
            od_hat[i,i] = 0
        od_hat[od_hat < 0] = 0
        od_hat = np.floor(od_hat)

        # logger.save_generated_od(geoids[idx], od_hat)
        # logger.plot_generated_od(geoids[idx], od_hat)

        # denoising process
        denoising_seq_thisone = denoising_seq[:, l:r, l:r]
        denoising_seq_thisone = scalers["od"].inverse_transform(denoising_seq_thisone.reshape([-1, 1])).reshape([denoising_seq_thisone.shape[0], denoising_seq_thisone.shape[1], denoising_seq_thisone.shape[2]])
        denoising_seq_thisone = scalers["od_normer"].inverse_transform(denoising_seq_thisone)
        for i in range(denoising_seq_thisone.shape[1]): # consistant with groundtruth
            denoising_seq_thisone[:, i,i] = 0
        denoising_seq_thisone[denoising_seq_thisone < 0] = 0
        denoising_seq_thisone = np.floor(denoising_seq_thisone)
        # logger.plot_denoising_process(geoids[idx], denoising_seq_thisone)

        # metrics
        one_metrics = cal_od_metrics(od_hat, od)
        all_metrics.append(one_metrics)

    return all_metrics
