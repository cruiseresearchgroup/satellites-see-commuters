import time
import random
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from pprint import pprint

from utils.tool import *
from utils.metrics import *
from data_load import *


def train(config, pre_model, diff_model, train_Dloader, valid_Dloader, logger, scalers):
    optm = torch.optim.AdamW(diff_model.model.parameters(), lr=config["learning_rate"], weight_decay=0.001)
    scheduler = StepLR(optm, step_size=5000, gamma=0.9)
    if config["mode"] == "CONTINUE":
        path = logger.optimizer_path[:-4] + "_lastest.pkl"
        state_dict = torch.load(path, map_location=config["device"])
        optm.load_state_dict(state_dict)
        start_epochs = logger.exp_log["train_log"]["num_epochs"]
    else:
        start_epochs = 0

    print("--------------------------------------------------")
    Ts = [x for x in range(config["T"])]
    for epoch in range(start_epochs, config["EPOCH"]):
        ###################### train
        diff_model.train()
        print("  ** Training. Epoch:", epoch)
        train_loss = 0
        for i, batch in tqdm(enumerate(train_Dloader), total=len(train_Dloader), unit="batch"):
            _, batched_graph, batched_dis, ods = batch

            optm.zero_grad()

            # t
            t = torch.LongTensor([random.choice(Ts) for _ in ods]).to(config["device"])
            # n
            n = batched_graph.ndata["nfeat"].to(config["device"])
            # e
            dim = sum([x.shape[0] for x in ods])
            batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
            batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"]) # There are zero-noise from blocking the OD matrix.
            noiser_factors = []
            l, r = 0, 0
            for i, od in enumerate(ods):
                noiser_factor = torch.rand(1)
                noiser_factors.extend([noiser_factor for _ in range(od.shape[0])])
                l = r
                r = r + od.shape[0]
                batched_od[l:r, l:r] = ods[i]
                batchlization[l:r, l:r] = 1
            noiser_factors = torch.stack(noiser_factors).to(config["device"])
            e = batched_od.to(config["device"])
            # img introduce
            n_traffic = batched_graph.ndata["trafeat"].to(config["device"]) if "trafeat" in batched_graph.ndata else None
            n_attr = n[:, 0: config["n_indim"]]
            noisy_attr = n_attr * noiser_factors + torch.randn_like(n_attr) * (1 - noiser_factors)
            n_img = n[:, config["n_indim"]: ]
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
            # mask
            mask_n, mask_e = generate_masks(config, n, e)
            mask_n, mask_e = mask_n.to(config["device"]), mask_e.to(config["device"])
            masks = (mask_n, mask_e)
            # loss
            loss = diff_model.loss(net, masks, t, batched_dis.to(config["device"]), batchlization)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.)
            optm.step()
        scheduler.step()
        train_loss = train_loss / len(train_Dloader)
        logger.once_train_record(epoch, train_loss, diff_model, optm)
        print("--------------------------------------------------")

        ###################### valid
        diff_model.eval()
        if (epoch % config["valid_period"] == 0) and (epoch != 0):
            print("--------------------------------------------------")
            with torch.no_grad():
                print("  ** Valid. Epoch:", epoch)
                valid_evaluator = validEvaluator(config)
                valid_loss = 0
                for i, batch in tqdm(enumerate(valid_Dloader), total=len(valid_Dloader), unit="batch"):
                    _, batched_graph, batched_dis, ods = batch

                    # t
                    t = torch.LongTensor([random.choice(Ts)]).to(config["device"])
                    # n
                    n = batched_graph.ndata["nfeat"].to(config["device"])
                    # e
                    dim = sum([x.shape[0] for x in ods])
                    batched_od = torch.full((dim, dim), 0, dtype=torch.float32)
                    batchlization = torch.full((dim, dim), 0, dtype=torch.float32).to(config["device"]) # There are zero-noise from blocking the OD matrix.
                    noiser_factors = []
                    l, r = 0, 0
                    for i, od in enumerate(ods):
                        noiser_factor = torch.rand(1)
                        noiser_factors.extend([noiser_factor for _ in range(od.shape[0])])
                        l = r
                        r = r + od.shape[0]
                        batched_od[l:r, l:r] = ods[i]
                        batchlization[l:r, l:r] = 1
                    noiser_factors = torch.stack(noiser_factors).to(config["device"])
                    e = batched_od.to(config["device"])
                    # img introduce
                    n_traffic = batched_graph.ndata["trafeat"].to(config["device"]) if "trafeat" in batched_graph.ndata else None
                    n_attr = n[:, 0: config["n_indim"]]
                    noisy_attr = n_attr * noiser_factors + torch.randn_like(n_attr) * (1 - noiser_factors)
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
                    # mask
                    mask_n, mask_e = generate_masks(config, n, e)
                    mask_n, mask_e = mask_n.to(config["device"]), mask_e.to(config["device"])
                    masks = (mask_n, mask_e)

                    # loss
                    loss = diff_model.loss(net, masks, t, batched_dis, batchlization)
                    valid_loss += loss.item()

                    # metrics
                    mask_n, mask_e, flags = generate_eval_mask(config, n, e)
                    mask_n, mask_e = mask_n.to(config["device"]), mask_e.to(config["device"])
                    masks = (mask_n, mask_e)
                    c = (net, masks, batched_dis, batchlization)
                    # generate
                    n_hats, e_hats = [], []
                    for _ in range(config["sample_times"]):
                        net_hat = diff_model.DDIM_sample_loop(n.shape, e.shape, c)[-1]
                        n_hat, e_hat = net_hat
                        n_hats.append(n_hat)
                        e_hats.append(e_hat)
                    n_hat = torch.mean(torch.stack(n_hats), dim=0)
                    e_hat = torch.mean(torch.stack(e_hats), dim=0)

                    # eval
                    pred = (n_hat.cpu().numpy(), e_hat.cpu().numpy())
                    masks = (mask_n.cpu().numpy(), mask_e.cpu().numpy())
                    valid_evaluator.evaluate_batch(pred=pred,
                                                   gt=(n, ods),
                                                   masks=masks,
                                                   flags=flags,
                                                   scalers=scalers)
                valid_loss = valid_loss / len(valid_Dloader)
                metrics = valid_evaluator.summary_all_metrics()

                logger.once_valid_record(epoch, valid_loss, metrics, diff_model, optm)
            logger.save_exp_log()
            print("--------------------------------------------------")
            if logger.overfit_flag >= config["overfit_tolerance"]:
                print(" ** Early stop!")
                print("--------------------------------------------------")
                break


def pre_train_NN(config, pre_model, train_Dloader, validSet, logger, scalers):
    # collect data of all OD pairs
    print("\n  ** Pre-training NN...")
    print("--------------------------------------------------")

    optm = torch.optim.AdamW(pre_model.parameters(), lr=3e-4, weight_decay=0.001)
    for epoch in range(0, 10000):
        # train
        pre_model.train()
        t_losses = []
        for i, batch in tqdm(enumerate(train_Dloader), total=len(train_Dloader), unit="batch"):
            x, y = batch
            optm.zero_grad()
            y_hat = pre_model(x.to(config["device"]))
            loss = F.mse_loss(y_hat, y.to(config["device"]))
            t_losses.append(loss.item())
            loss.backward()
            optm.step()
        t_loss = np.mean(t_losses)

        # valid
        pre_model.eval()
        with torch.no_grad():
            all_metrics = []
            v_losses = []
            for one in tqdm(validSet, desc="eval."):
                feat = one["nfeat"]
                dis = one["dis"]
                od = one["od"]
                feat, _ = net_to_pairs(feat, dis, od)
                feat = torch.FloatTensor(feat).to(config["device"])

                od_hat = pre_model(feat)
                valid_loss = F.mse_loss(od_hat.reshape([-1]), torch.FloatTensor(od).reshape([-1]).to(config["device"]))
                v_losses.append(valid_loss.item())
                od_hat = od_hat.reshape(od.shape).cpu().numpy()
                od = scalers["od"].inverse_transform(od)
                od_hat = scalers["od"].inverse_transform(od_hat)
                od = scalers["od_normer"].inverse_transform(od)
                od_hat = scalers["od_normer"].inverse_transform(od_hat)
                od_hat[od_hat < 0] = 0
                metrics = cal_od_metrics(od_hat, od)
                all_metrics.append(metrics)
            v_loss = np.mean(v_losses)

            logger.pretrain_loss_record(t_loss, v_loss, epoch)

            sums = defaultdict(float)
            for d in all_metrics:
                for key, value in d.items():
                    sums[key] += value
            averages = {key: value / len(all_metrics) for key, value in sums.items()}
            SEG_metrics = citywise_segmented_metrics(all_metrics)
            SEG_metrics["all"] = averages
            logger.pretrain_metrics_record(SEG_metrics, epoch)
        
        # save model
        torch.save(pre_model.state_dict(), "exp/model/pretrain/"+config["exp_name"]+"_"+str(epoch)+".pkl")
            

def pretrain_denoiser(config, pre_model, train_Dloader, validCities, valid_Dloader, logger, scalers):
    print("\n  ** Pre-training denoiser...")
    print("--------------------------------------------------")

    patience = 1000
    best_loss = np.inf
    best_losses_at_different_p = [np.inf for _ in range(5)]
    epochs_no_improve = 0
    optm = torch.optim.AdamW(pre_model.parameters(), lr=3e-4)
    for epoch in range(0, int(9e9)):
        starttime = time.time()
        # train
        pre_model.train()
        t_losses = []
        for _, batch in enumerate(train_Dloader):
            imgfeat, noisy_attr, attr = batch
            optm.zero_grad()
            pred_attr = pre_model(imgfeat.to(config["device"]), noisy_attr.to(config["device"]))
            loss = F.mse_loss(pred_attr, attr.to(config["device"]))
            # pred_pop = pre_model(imgfeat.to(config["device"]))
            # loss = F.mse_loss(pred_pop, attr.to(config["device"])[:, 0])
            t_losses.append(loss.item())
            loss.backward()
            optm.step()
        t_loss = np.mean(t_losses)

        # valid
        pre_model.eval()
        with torch.no_grad():
            v_losses = []
            for _ in range(10):
                for _, batch in enumerate(valid_Dloader):
                    imgfeat, noisy_attr, attr = batch
                    pred_attr = pre_model(imgfeat.to(config["device"]), noisy_attr.to(config["device"]))
                    valid_loss = F.mse_loss(pred_attr, attr.to(config["device"]))
                    # pred_pop = pre_model(imgfeat.to(config["device"]))
                    # valid_loss = F.mse_loss(pred_pop, attr.to(config["device"])[:, 0])
                    v_losses.append(valid_loss.item())
            v_loss = np.mean(v_losses)
        
        if v_loss < best_loss:
            best_loss = v_loss
            epochs_no_improve = 0
             # # save model
            os.makedirs("exp/model/pretrain/", exist_ok=True)
            torch.save(pre_model.state_dict(), "exp/model/pretrain/"+config["exp_name"]+".pkl")
            for idx, p in enumerate([0, 0.25, 0.5, 0.75, 1]):
                config_cur = deepcopy(config)
                config_cur["p_noisyAttr"] = p
                valid_ds_cur = NAE_dataset(validCities, config_cur)
                valid_dl_cur = DataLoader(valid_ds_cur, batch_size=len(valid_ds_cur))
                for _, batch in enumerate(valid_dl_cur):
                    imgfeat, noisy_attr, attr = batch
                    pred_attr = pre_model(imgfeat.to(config["device"]), noisy_attr.to(config["device"]))
                    valid_loss_cur = F.mse_loss(pred_attr, attr.to(config["device"]))
                best_losses_at_different_p[idx] = valid_loss_cur.item()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping triggered! the best v_loss=", best_loss)
            import pandas as pd
            df = pd.DataFrame({"p": [0, 0.25, 0.5, 0.75, 1], "loss": best_losses_at_different_p})
            print(df)
            exit()
            
        print("  ** Epoch: {} train loss: {:.6e} valid loss: {:.6e} consume: {:4.2f}s.".format(epoch, t_loss, v_loss, time.time()-starttime))
