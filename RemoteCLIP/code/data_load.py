import os
from random import shuffle, choice

import pickle as pkl
import numpy as np

import torch
import dgl

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from utils.tool import *


def prepare_big(config, scalers):
    cities = os.listdir(config["data_path"])
    if config["shuffle_cities"] == 1:
        shuffle(cities)

    # load data
    print("***************** load data *****************")
    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "nfeat": np.load(config["data_path"] + city + "/nfeat/nfeat.npy"),
                "dis": np.load(config["data_path"] + city + "/adj/dis.npy"),
                "od": scalers["od_normer"].fit_transform(np.load(config["data_path"] + city + "/od/od.npy"))
            }
        if 700 < one["nfeat"].shape[0] < 1000:
            for i in range(one["od"].shape[0]):
                one["od"][i,i] = 0
            data.append(one)
    
    # normalization
    print("  ** normalizing dataset...", end="")
    if config["attr_MinMax"] == 1:
        for i, nfeat in enumerate([x["nfeat"] for x in data]):
            data[i]["nfeat"] = scalers["feat"].transform(nfeat)
        for i, dis in enumerate([x["dis"] for x in data]):
            data[i]["dis"] = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])

    if config["od_MinMax"] == 1:
        for i, od in enumerate([x["od"] for x in data]):
            data[i]["od"] = scalers["od"].transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
    print("done")
    
    return data

def prepare_data(config):

    if config["skew_norm"] == "none":
        OD_normer = None_transformer()
    elif config["skew_norm"] == "log":
        OD_normer = Log_transformer()
    elif config["skew_norm"] == "boxcox":
        OD_normer = BoxCox_transformer()
    else:
        raise Exception("Unknown skew_norm type")


    cities = os.listdir(config["data_path"])
    if config["shuffle_cities"] == 1:
        shuffle(cities)

    # load data
    print("***************** load data *****************")
    # worldpop part
    worldpop = np.load("data/worldpop.npy")
    worldpop = worldpop[:, np.argsort(worldpop[0])]

    data = []
    for city in tqdm(cities):
        one = {
                "GEOID" : city,
                "nfeat": np.load(config["data_path"] + city + "/nfeat/nfeat.npy"),
                "imgfeat" : np.load(config["data_path"] + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(config["data_path"] + city + "/adj/dis.npy"),
                "od": OD_normer.fit_transform(np.load(config["data_path"] + city + "/od/od.npy"))
            }
        if "~" in config["city_type_limit"]:
            if config["city_type_limit"][0] == "~":
                if (one["nfeat"].shape[0] > int(config["city_type_limit"][1:])):
                    continue
            elif config["city_type_limit"][-1] == "~":
                if (one["nfeat"].shape[0] <= int(config["city_type_limit"][:-1])):
                    continue
            else:
                low = config["city_type_limit"].split("~")[0]
                high = config["city_type_limit"].split("~")[1]
                if (one["nfeat"].shape[0] < int(low)) or (one["nfeat"].shape[0] > int(high)):
                    continue
        for i in range(one["od"].shape[0]):
            one["od"][i,i] = 0

        worldpop_feat = worldpop[:, (worldpop[0] // 1e6) == int(city)][1:, :].transpose()
        if worldpop_feat.shape[0] != one["imgfeat"].shape[0]:
            continue
        one["imgfeat"] = np.concatenate([one["imgfeat"], np.log1p(worldpop_feat)], axis=1)
        if config.get("use_trafeat", 0) == 1:
            trafeat_fname = config.get("trafeat_file", "trafeat.npy")
            trafeat_dim = config.get("trafeat_fallback_dim", 5)
            trafeat_path = config["data_path"] + city + "/nfeat/" + trafeat_fname
            if os.path.exists(trafeat_path):
                tf = np.load(trafeat_path)
                if config.get("trafeat_log1p", 0) == 1:
                    tf = np.log1p(tf)
                one["trafeat"] = tf
            else:
                one["trafeat"] = np.zeros((one["nfeat"].shape[0], trafeat_dim), dtype=np.float32)
        data.append(one)
    print("  ** constructing dataset...", end="")
    if config["split_TVT_geo"] == 1:
        train, valid, test = split_data_intoTVT_geo(data, config)
    elif config["split_TVT_geo"] == 0:
        train, valid, test = split_data_intoTVT(data, config)
    print("done, totally", len(data), "cities")

    # normalization
    print("  ** normalizing dataset...", end="")
    if config["attr_MinMax"] == 1:
        scaler_feat = MinMaxer(np.concatenate([x["nfeat"] for x in train], axis=0))
        for i, nfeat in enumerate([x["nfeat"] for x in train]):
            train[i]["nfeat"] = scaler_feat.transform(nfeat)
        for i, nfeat in enumerate([x["nfeat"] for x in valid]):
            valid[i]["nfeat"] = scaler_feat.transform(nfeat)
        for i, nfeat in enumerate([x["nfeat"] for x in test]):
            test[i]["nfeat"] = scaler_feat.transform(nfeat)
        scaler_dis = MinMaxer(np.concatenate([x["dis"].reshape([-1, 1]) for x in train], axis=0))
        for i, dis in enumerate([x["dis"] for x in train]):
            train[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
        for i, dis in enumerate([x["dis"] for x in valid]):
            valid[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
        for i, dis in enumerate([x["dis"] for x in test]):
            test[i]["dis"] = scaler_dis.transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    else:
        scaler_feat = None
        scaler_dis = None

    if config["od_MinMax"] == 1:
        scaler_od = MinMaxer(np.concatenate([x["od"].reshape([-1, 1]) for x in train], axis=0))
        for i, od in enumerate([x["od"] for x in train]):
            train[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        for i, od in enumerate([x["od"] for x in valid]):
            valid[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
        for i, od in enumerate([x["od"] for x in test]):
            test[i]["od"] = scaler_od.transform(od.reshape([-1, 1])).reshape([od.shape[0], od.shape[1]])
    else:
        scaler_od = None
    scalers = {
        "feat" : scaler_feat,
        "dis" : scaler_dis,
        "od" : scaler_od,
        "od_normer" : OD_normer,
    }
    print("done")
    
    return data, train, valid, test, scalers


def prepare_specific_dataloader(config, scalers):

    test_cities_dir = config.get("test_specific_dir", "TEST_UK")

    cities = os.listdir(f"data/{test_cities_dir}/")

    # load data
    print("\n***************** load specific data *****************")
    # worldpop part
    worldpop = np.load("data/worldpop.npy")
    worldpop = worldpop[:, np.argsort(worldpop[0])]

    data = []
    for city in tqdm(cities):
        raw_od = np.load(f"data/{test_cities_dir}/" + city + "/od/od.npy")
        for i in range(raw_od.shape[0]):
            raw_od[i, i] = 0
        log_od = scalers["od_normer"].fit_transform(raw_od)
        norm_od = scalers["od"].transform(log_od.reshape([-1, 1])).reshape(raw_od.shape)
        one = {
                "GEOID" : city,
                "imgfeat" : np.load(f"data/{test_cities_dir}/" + city + "/nfeat/imgfeat.npy"),
                "dis": np.load(f"data/{test_cities_dir}/" + city + "/adj/dis.npy"),
                "od": norm_od
            }

        worldpop_feat = np.load(f"data/{test_cities_dir}/" + city + "/worldpop.npy")
        one["imgfeat"] = np.concatenate([one["imgfeat"], np.log1p(worldpop_feat)], axis=1)
        if config.get("use_trafeat", 0) == 1:
            trafeat_fname = config.get("trafeat_file", "trafeat.npy")
            trafeat_dim = config.get("trafeat_fallback_dim", 5)
            trafeat_path = f"data/{test_cities_dir}/" + city + "/nfeat/" + trafeat_fname
            if os.path.exists(trafeat_path):
                tf = np.load(trafeat_path)
                if config.get("trafeat_log1p", 0) == 1:
                    tf = np.log1p(tf)
                one["trafeat"] = tf
            else:
                one["trafeat"] = np.zeros((one["imgfeat"].shape[0], trafeat_dim), dtype=np.float32)
        data.append(one)

    # normalization
    print("  ** normalizing specific dataset...", end="")
    if config["attr_MinMax"] == 1:
        for i, dis in enumerate([x["dis"] for x in data]):
            data[i]["dis"] = scalers["dis"].transform(dis.reshape([-1, 1])).reshape([dis.shape[0], dis.shape[1]])
    
    test_set = SpecificDataset(data, config, "test")
    ttBS = MyBatchSampler(test_set, 1, 10000)
    from dgl.dataloading import GraphDataLoader
    test_Dloader = GraphDataLoader(test_set, batch_sampler=ttBS, collate_fn=collate_fn)

    return test_Dloader


class MyDataset(dgl.data.DGLDataset):
    def __init__(self, data, config, Type):
        self.config = config
        self.data = data
        self.type = Type
        super(MyDataset,self).__init__(name="OD_graph_dataset")

    def process(self):
        self.GEOIDs = []
        self.graphs = []
        self.dises = []
        self.ods = []
        for one in self.data:
            self.GEOIDs.append(one["GEOID"])
            g = dgl.graph(one["od"].nonzero(), num_nodes=one["od"].shape[0])
            g.ndata["nfeat"] = torch.FloatTensor(np.concatenate([one["nfeat"], one["imgfeat"]], axis=1))
            if "trafeat" in one:
                g.ndata["trafeat"] = torch.FloatTensor(one["trafeat"])
            self.graphs.append(g)
            self.dises.append(torch.FloatTensor(one["dis"]))
            self.ods.append(torch.FloatTensor(one["od"]))

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if self.type != "train":
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]
        if self.config["pert_node"] == 1:
            geoid = self.GEOIDs[index]
            graph = self.graphs[index]
            dis = self.dises[index]
            od = self.ods[index]
            graph, dis, od = permut_nodes_order(graph, dis, od)
            return geoid, graph, dis, od
        else:
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]

    
class SpecificDataset(dgl.data.DGLDataset):
    def __init__(self, data, config, Type):
        self.config = config
        self.data = data
        self.type = Type
        super(SpecificDataset,self).__init__(name="SpecificDataset")

    def process(self):
        self.GEOIDs = []
        self.graphs = []
        self.dises = []
        self.ods = []
        for one in self.data:
            self.GEOIDs.append(one["GEOID"])
            g = dgl.graph(one["od"].nonzero(), num_nodes=one["od"].shape[0])
            g.ndata["nfeat"] = torch.FloatTensor(one["imgfeat"])
            if "trafeat" in one:
                g.ndata["trafeat"] = torch.FloatTensor(one["trafeat"])
            self.graphs.append(g)
            self.dises.append(torch.FloatTensor(one["dis"]))
            self.ods.append(torch.FloatTensor(one["od"]))

    def get_size(self, index):
        return self.dises[index].shape[0]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        if self.type != "train":
            return self.GEOIDs[index], self.graphs[index], self.dises[index], self.ods[index]



class MyBatchSampler:
    def __init__(self, dataset, batch_size, max_value):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_value = max_value

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                yield batch

        if batch:
            yield batch

    # def __len__(self):
    #     return len(self.dataset) // self.batch_size + 1
    
    def __len__(self):
        indices = list(range(len(self.dataset)))
        count = 0

        while indices:
            batch = []
            current_sum = 0
            for _ in range(len(indices)):
                index = choice(indices)
                item_size = self.dataset.get_size(index)
                if (current_sum + item_size <= self.max_value) and (len(batch) < self.batch_size):
                    batch.append(index)
                    current_sum += item_size
                    indices.remove(index)
                else:
                    break

            if len(batch) > 0:
                count += 1

        return count
    

class PreTrain_dataset(torch.utils.data.Dataset):
    def __init__(self, cities, config):
        self.config = config

        self.get_all_pairs(cities)

    def get_all_pairs(self, cities):
        feats = []
        ods = []
        for city in cities:
            o_feat = city["nfeat"][:, np.newaxis, :].repeat(city["nfeat"].shape[0], axis=1)
            d_feat = city["nfeat"][np.newaxis, :, :].repeat(city["nfeat"].shape[0], axis=0)
            dis = city["dis"][:, :, np.newaxis]
            feat = np.concatenate([o_feat, d_feat, dis], axis=2)
            feat = feat.reshape([-1, feat.shape[2]])
            od = city["od"].reshape([-1, 1])
            if feat.shape[1] != 195:
                print(feat.shape, od.shape)
            feats.append(feat)
            ods.append(od)
        x = np.concatenate(feats, axis=0)
        y = np.concatenate(ods, axis=0)
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

class NAE_dataset(torch.utils.data.Dataset):
    def __init__(self, cities, config):
        self.config = config
        
        self.getImgfeatAllregions(cities)
        self.getAttrfeatAllregions(cities)

    def getAttrfeatAllregions(self, cities):
        attrfeats = []
        for city in cities:
            attrfeat = city["nfeat"]
            attrfeats.append(attrfeat)
        if self.config["if-logscaleEval"] == 0:
            self.attrfeats = torch.FloatTensor(np.concatenate(attrfeats, axis=0))
        elif self.config["if-logscaleEval"] == 1:
            self.attrfeats = torch.log1p(torch.FloatTensor(np.concatenate(attrfeats, axis=0)))

        # noisy
        if self.config["p_noisyAttr"] == -1:
            weights = torch.rand(self.attrfeats.shape[0], 1)
        else:
            weights = self.config["p_noisyAttr"]
        noise = torch.randn(self.attrfeats.shape[0], self.attrfeats.shape[1])
        self.noisy_attr = self.attrfeats * (1 - weights) + noise * weights

    def getImgfeatAllregions(self, cities):
        imgfeats = []
        for city in cities:
            imgfeat = city["imgfeat"]
            imgfeats.append(imgfeat)
        self.imgfeats = torch.FloatTensor(np.concatenate(imgfeats, axis=0))

    def __len__(self):
        return len(self.imgfeats)
    
    def __getitem__(self, index):
        return self.imgfeats[index] if self.config["if_imgAugDenosing"] else self.imgfeats[index] *0, self.noisy_attr[index], self.attrfeats[index]
