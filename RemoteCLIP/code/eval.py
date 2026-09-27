import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import torch

from tqdm import tqdm
from pprint import pprint

from utils.tool import generate_Gen_mask, mean_the_denoising_process
from utils.metrics import *

# fixed json keys for the noise levels (avoid mixing "1" and "1.0")
NOISE_LEVELS = [0, 0.25, 0.5, 0.75, 1]
_KEY = {float(n): ("%g" % n) for n in NOISE_LEVELS}


def _levels():
    """Noise levels to run. Set ONLY_NOISE to run a single level."""
    only = os.environ.get("ONLY_NOISE", "").strip()
    if only:
        v = float(only)
        return [int(v) if v in (0.0, 1.0) else v]
    return list(NOISE_LEVELS)


def _merge_save(logger, key, eval_metrics):
    """Merge new results into the existing log before saving."""
    if os.path.exists(logger.exp_path):
        with open(logger.exp_path, "r") as f:
            log = json.load(f)
    else:
        log = {"train_log": {}}
    prev = {float(k): v for k, v in (log.get(key) or {}).items()}
    prev.update(eval_metrics)
    log[key] = _ser(prev)
    with open(logger.exp_path, "w") as f:
        json.dump(log, f, indent=4)
    return len(prev)


def _ser(metrics):
    """Convert float keys to the fixed string keys."""
    return {_KEY.get(float(k), "%g" % float(k)): v for k, v in metrics.items()}


def test_specific(config, pre_model, diff_model, test_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Evaluation. :")

        # Load existing results to skip completed noise levels
        eval_metrics = {}
        if os.path.exists(logger.exp_path):
            with open(logger.exp_path, "r") as f:
                log = json.load(f)
            eval_metrics = {float(k): v for k, v in log.get("eval_metrics_specific", {}).items()}

        ckpt_dir = os.path.join(config["exp_path"], "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        for noiser_factor in _levels():
            if noiser_factor in eval_metrics:
                print("  noise=%s already done, skipping" % noiser_factor)
                continue

            gen_dir = logger.generation_directory + "UK_noise" + str(noiser_factor) + "/"
            os.makedirs(gen_dir, exist_ok=True)

            # Per-city checkpoint for this noise level
            ckpt_path = os.path.join(ckpt_dir, "%s_n%s_partial.pkl" % (config["exp_name"], noiser_factor))
            Gen_metrics = []
            done_cities = set()
            if os.path.exists(ckpt_path):
                # corrupted checkpoint (e.g. walltime kill), delete it and restart the level
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    Gen_metrics = ckpt["metrics"]
                    done_cities = set(ckpt["done_cities"])
                    print("  Resuming noise=%s from checkpoint (%d cities done)" % (noiser_factor, len(done_cities)))
                except Exception as exc:
                    print("  CORRUPT checkpoint %s (%s), discarding and restarting this level" % (ckpt_path, exc))
                    Gen_metrics = []
                    done_cities = set()
                    try:
                        os.remove(ckpt_path)
                    except OSError:
                        pass

            for i, batch in tqdm(enumerate(test_Dloader), total=len(test_Dloader), desc="noise-"+str(noiser_factor)):
                geoid, batched_graph, batched_dis, ods = batch
                if geoid[0] in done_cities:
                    continue
                # n
                n = batched_graph.ndata["nfeat"].to(config["device"])
                # e
                dim = sum([x.shape[0] for x in ods])
                batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"])
                l, r = 0, 0
                for i, od in enumerate(ods):
                    l = r
                    r = r + od.shape[0]
                    batched_od[l:r, l:r] = ods[i]
                    batchlization[l:r, l:r] = 1
                e = batched_od.to(config["device"])
                # img introduce
                n_traffic = batched_graph.ndata["trafeat"].to(config["device"]) if "trafeat" in batched_graph.ndata else None
                n_attr = n[:, 0: config["n_indim"]]
                noisy_attr = n_attr * noiser_factor + torch.randn_like(n_attr) * (1 - noiser_factor)
                n_img = n
                with torch.no_grad():
                    pred_attr = pre_model(n_img, noisy_attr)
                if config["if-ImgAttrAug"] == 0:
                    pred_attr = pred_attr * 0
                parts = [n_img, noisy_attr, pred_attr]
                if n_traffic is not None:
                    parts.append(n_traffic)
                n = torch.cat(parts, dim=1)
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
                for i in range(od_hat.shape[0]):
                    od_hat[i, i] = 0
                od_hat[od_hat < 0] = 0
                od_hat = np.floor(od_hat)
                np.save(gen_dir + geoid[0] + ".npy", od_hat)
                Gen_metrics.extend(eval_gen(config, logger, geoid, ods, diff_model, net, batched_dis, batchlization, scalers))

                # Save per-city checkpoint
                done_cities.add(geoid[0])
                # write to a temp file first so a walltime kill cannot corrupt the checkpoint
                _tmp = "%s.tmp%d" % (ckpt_path, os.getpid())
                torch.save({"metrics": Gen_metrics, "done_cities": list(done_cities)}, _tmp)
                os.replace(_tmp, ckpt_path)

            eval_metrics_p = average_listed_metrics(Gen_metrics)
            eval_metrics[noiser_factor] = eval_metrics_p

            # Clean up checkpoint for completed noise level
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

            # Save to JSON after each noise level
            n_have = _merge_save(logger, "eval_metrics_specific", eval_metrics)
            print("  saved noise=%s -> %d/5 levels in log" % (noiser_factor, n_have))

        pprint(eval_metrics)
    print("--------------------------------------------------")


def test(config, pre_model, diff_model, test_Dloader, logger, scalers):
    print("--------------------------------------------------")
    with torch.no_grad():
        print("  ** Evaluation. :")

        # resume from existing log, same as test_specific()
        eval_metrics = {}
        if os.path.exists(logger.exp_path):
            with open(logger.exp_path, "r") as f:
                log = json.load(f)
            # json keys are strings, convert to float
            eval_metrics = {float(k): v for k, v in log.get("eval_metrics", {}).items()}
            if eval_metrics:
                print("  resuming: %d/5 oise levels already done %s"
                      % (len(eval_metrics), sorted(eval_metrics)))

        for noiser_factor in _levels():
            if noiser_factor in eval_metrics:
                print("  noise=%s already done, skipping" % noiser_factor)
                continue
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
                n_traffic = batched_graph.ndata["trafeat"].to(config["device"]) if "trafeat" in batched_graph.ndata else None
                n_attr = n[:, 0: config["n_indim"]]
                noisy_attr = n_attr * noiser_factor + torch.randn_like(n_attr) * (1 - noiser_factor)
                n_img = n[:, config["n_indim"]:]
                with torch.no_grad():
                    pred_attr = pre_model(n_img, noisy_attr)
                if config["if-ImgAttrAug"] == 0:
                    pred_attr = pred_attr *0
                parts = [n_img, noisy_attr, pred_attr]
                if n_traffic is not None:
                    parts.append(n_traffic)
                n = torch.cat(parts, dim=1)
                # net
                net = (n, e)
                # dis
                batched_dis = batched_dis.to(config["device"])

                # metrics of one batch
                Gen_metrics.extend(eval_gen(config, logger, geoids, ods, diff_model, net, batched_dis, batchlization, scalers))
            
            # summary metrics and save log
            eval_metrics_p = average_listed_metrics(Gen_metrics)
            eval_metrics[noiser_factor] = eval_metrics_p
            # save after each noise level
            n_have = _merge_save(logger, "eval_metrics", eval_metrics)
            print("  saved noise=%s -> %d/5 levels complete" % (noiser_factor, n_have))
        pprint(eval_metrics)

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
