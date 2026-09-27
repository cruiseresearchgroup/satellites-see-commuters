from setproctitle import setproctitle

from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader

from train import *
from eval import test, test_specific
from utils.procedure import *
from utils.tool import collate_fn
from data_load import *
from model import *
from utils.MyLogger import Logger


def main(config):
    # meta
    print("\n****** experiment name:", config["exp_name"], " ******\n")
    setproctitle(config["exp_name"] + "@rc")
    setRandomSeed(config["random_seed"])

    # load data
    data, trainSet, validSet, testSet, scalers = prepare_data(config)
    # validSet = prepare_big(config, scalers) # check small to big
    config["num_train_samples"] = len(trainSet)
    config["num_valid_samples"] = len(validSet)
    config["num_test_samples"] = len(testSet)
    
    # set input dim for model
    config["n_indim"] = data[0]["nfeat"].shape[1]
    config["e_indim"] = 2
    config["n_outdim"] = config["n_indim"]
    config["e_outdim"] = 1
    config["img_dim"] = data[0]["imgfeat"].shape[1]
    config["trafeat_dim"] = data[0]["trafeat"].shape[1] if "trafeat" in data[0] else 0

    print("  ** constructing dataloader...", end="")
    train_set = MyDataset(trainSet, config, "train")
    valid_set = MyDataset(validSet, config, "valid")
    test_set = MyDataset(testSet, config, "test")
    tnBS = MyBatchSampler(train_set, config["batch_size"], config["max_nodes"])
    vdBS = MyBatchSampler(valid_set, config["batch_size"], config["max_nodes"])
    ttBS = MyBatchSampler(test_set, config["batch_size"], config["max_nodes"])
    train_Dloader = GraphDataLoader(train_set, batch_sampler=tnBS, collate_fn=collate_fn)
    valid_Dloader = GraphDataLoader(valid_set, batch_sampler=vdBS, collate_fn=collate_fn)
    test_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)
    if config["test_specific"] == 1:
        test_Dloader = prepare_specific_dataloader(config, scalers)

    print("done")

    # logger
    print("  ** preparing logger...", end="")
    logger = Logger(config)
    if config["mode"] == "CONTINUE":
        logger.load_exp_log()
    print("done")

    if config["mode"] == "pretrainDenoiser":
        # data
        pre_train_ds = NAE_dataset(trainSet, config)
        pre_train_dl = DataLoader(pre_train_ds, batch_size=1024, shuffle=True)
        valid_ds = NAE_dataset(validSet, config)
        valid_dl = DataLoader(valid_ds, batch_size=len(valid_ds))
        # model
        # pre_model = Noisy_AutoEncoder(config).to(config["device"])
        pre_model = DenoisingTransformer(config).to(config["device"])
        # pre_model = CheckPrepopMLP(config).to(config["device"])
        # train
        pretrain_denoiser(config, pre_model, pre_train_dl, validSet, valid_dl, logger, scalers)
    else:
        pre_model = DenoisingTransformer(config).to(config["device"])
        # Attention!!!! need to replace the pretrain model by the corresponding one manully!!!
        pre_model.load_state_dict(torch.load("exp/model/pretrain/pretrain.pkl", map_location=config["device"]))

    # model
    print("  ** preparing model...", end="")
    diff_model = Diffusion(config).to(config["device"])
    if config["mode"] == "CONTINUE":
        path = logger.model_path[:-4] + "_lastest.pkl"
        state_dict = torch.load(path, map_location=config["device"])
        diff_model.load_state_dict(state_dict)
    print("done \n")
    
    if config["mode"] != "EVAL":
        # train
        train(config, pre_model, diff_model, train_Dloader, valid_Dloader, logger, scalers)

    # test
    # load best valid model first (model_exp_name allows loading from a different experiment)
    model_exp_name = config.get("model_exp_name", config["exp_name"])
    model_best_path = config["exp_path"] + "model/" + model_exp_name + "/model_" + str(config["random_seed"]) + "_best.pkl"
    diff_model.load_state_dict(torch.load(model_best_path, map_location=config["device"]))
    if config["test_specific"] == 1:
        test_specific(config, pre_model, diff_model, test_Dloader, logger, scalers)
    else:
        test(config, pre_model, diff_model, test_Dloader, logger, scalers)

    print("  ** Finished!")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config = get_config(args.config)
    main(config)

    #################################################################################################################################

    # # check scale vs. performance train
    # hostname = socket.gethostname()

    # args = add_args()

    # layer = args.layer
    # hiddim = args.hiddim
    # num_city = args.cities

    # def run_experiment(config):
    #     redirect_output_to_file(config['exp_name'])
    #     main(config)

    # with open("exp/config/us.json") as f:
    #     config = json.load(f)
    # config["exp_name"] = config["exp_name"] + "_L" + str(layer) + "H" + str(hiddim) + "C" + str(num_city)

    # if os.path.exists("exp/log/"+config["exp_name"]+"_"+str(config["random_seed"])+".json"):
    #     with open("exp/log/"+config["exp_name"]+"_"+str(config["random_seed"])+".json") as f:
    #         log = json.load(f)
    #         if "eval_metrics" in log.keys():
    #             print(config["exp_name"], "is already finised.")
    #             exit()

    # config["mem_need"] = int(config["mem_need"] * 0.95 * (hiddim / config["hiddim"]) * (layer / config["num_layer"]))
    # if hostname == "DL4":
    #     if config["mem_need"] > 23000:
    #         exit()

    # GPU = choose_gpu_from([0, 1, 2, 3, 4], mem_need = config["mem_need"])
    # config["device"] = torch.device(int(GPU))

    # config["train_set"] = num_city
    # config["valid_set"] = 220
    # config["test_set"] = 220

    # config["num_layer"] = layer
    # config["hiddim"] = hiddim

    # run_experiment(config)

    #################################################################################################################################

    # # check scale vs. performance test
    # hostname = socket.gethostname()

    # args = add_args()

    # layer = args.layer
    # hiddim = args.hiddim
    # num_city = args.cities

    # def run_experiment(config):
    #     redirect_output_to_file(config['exp_name'])
    #     main(config)

    # with open("exp/config/us.json") as f:
    #     config = json.load(f)
    # config["exp_name"] = config["exp_name"] + "_L" + str(layer) + "H" + str(hiddim) + "C" + str(num_city)

    # if os.path.exists("exp/model/"+config["exp_name"]+"/"):
    #     Flag = False
    #     for file in os.listdir("exp/model/"+config["exp_name"]+"/"):
    #         if file[-8:] == "best.pkl":
    #             Flag = True
    #             break
    #     if Flag:
    #         config["mem_need"] = int(config["mem_need"] * 0.95 * (hiddim / config["hiddim"]) * (layer / config["num_layer"]))
    #         if hostname == "DL4":
    #             if config["mem_need"] > 23000:
    #                 exit()

    #         GPU = choose_gpu_from([0, 1, 2, 3, 4], mem_need = config["mem_need"])
    #         config["device"] = torch.device(int(GPU))

    #         config["train_set"] = num_city
    #         config["valid_set"] = 220
    #         config["test_set"] = 220

    #         config["num_layer"] = layer
    #         config["hiddim"] = hiddim

    #         run_experiment(config)