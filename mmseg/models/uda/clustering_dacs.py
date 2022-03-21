import numpy as np
import torch

from mmseg.models import UDA
from mmseg.models.uda.dacs import DACS
import wandb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment


def get_best_match_aux(distss):
    n_clusters = len(distss)
    print('n_clusterss', n_clusters)
    res = linear_sum_assignment(distss)[1].tolist()
    targets = [None] * n_clusters
    for x, y in enumerate(res):
        targets[y] = x
    return targets


def get_best_match(sc, tc):
    dists = np.full((sc.shape[0], tc.shape[0]), fill_value=np.inf)
    for i in range(sc.shape[0]):
        for j in range(tc.shape[0]):
            dists[i][j] = np.mean((sc[i] - tc[j]) ** 2)
    best_match = get_best_match_aux(dists.copy())

    return best_match


@UDA.register_module()
class ClusteringDACS(DACS):
    def __init__(self, **cfg):
        wandb.init(
            project='DAFormer',
            id=wandb.util.generate_id(),
            name='temp1'
        )
        self.source_ds = cfg.pop('source_ds')
        self.target_ds = cfg.pop('target_ds')
        self.n_clusters = cfg.pop('n_clusters')
        self.acc_amount = cfg.pop('acc_amount')
        super(ClusteringDACS, self).__init__(**cfg)
        self.slice_to_cluster = None
        self.source_clusters = None
        self.target_clusters = None
        self.best_matchs = None
        self.best_matchs_indexes = None
        self.accumulate_for_loss = []
        self.slice_to_feature_source = {}
        self.slice_to_feature_target = {}
        self.avg_dist_losses = []
        self.step_counter = 0
        self.clustering_dacs = True

    def calc_align_loss(self, features1=None, img_metas=None, device='cpu'):
        self.step_counter += 1

        if self.step_counter == 300:
            self.step_counter = 0
            self.source_clusters = []
            self.target_clusters = []
            self.accumulate_for_loss = []
            for _ in range(self.n_clusters):
                self.accumulate_for_loss.append([])
                self.source_clusters.append([])
                self.target_clusters.append([])
            p = PCA(n_components=20, random_state=42)
            t = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
            points = []
            for img_name, feat in self.slice_to_feature_source.items():
                points.append(feat)
            for img_name, feat in self.slice_to_feature_target.items():
                points.append(feat)
            points = np.array(points)
            points = points.reshape(points.shape[0], -1)
            print('doing tsne')
            points = p.fit_transform(points)
            points = t.fit_transform(points)
            source_points, target_points = points[:len(self.slice_to_feature_source)], points[
                                                                                       len(self.slice_to_feature_source):]
            k1 = KMeans(n_clusters=self.n_clusters, random_state=42)
            print('doing kmean 1')
            sc = k1.fit_predict(source_points)
            k2 = KMeans(n_clusters=self.n_clusters, random_state=42, init=k1.cluster_centers_)
            print('doing kmean 2')
            tc = k2.fit_predict(target_points)
            print('getting best match')
            best_matchs_indexes = get_best_match(k1.cluster_centers_, k2.cluster_centers_)
            self.slice_to_cluster = {}
            items = list(self.slice_to_feature_source.items())
            for i in range(len(self.slice_to_feature_source)):
                self.source_clusters[sc[i]].append(items[i][1])
                self.slice_to_cluster[items[i][0]] = sc[i]
            items = list(self.slice_to_feature_target.items())
            for i in range(len(self.slice_to_feature_target)):
                self.slice_to_cluster[items[i][0]] = tc[i]
            for i in range(len(self.source_clusters)):
                self.source_clusters[i] = np.mean(self.source_clusters[i], axis=0)
            self.best_matchs = []
            for i in range(len(best_matchs_indexes)):
                self.best_matchs.append(torch.tensor(self.source_clusters[best_matchs_indexes[i]]))
            self.slice_to_feature_source = {}
            self.slice_to_feature_target = {}

        dist_loss = torch.tensor(0.0, device=device)
        for i in range(features1.shape[0]):
            feat1, meta1 = features1[i], img_metas[i]
            imname = meta1['ori_filename'].split('.')[0]
            if self.source_ds in meta1['filename']:
                self.slice_to_feature_source[imname] = feat1.detach().cpu().numpy()
            else:
                # assert self.target_ds in  meta1['filename']
                self.slice_to_feature_target[imname] = feat1.detach().cpu().numpy()
                if self.best_matchs is not None and imname in self.slice_to_cluster:
                    self.accumulate_for_loss[self.slice_to_cluster[imname]].append(feat1)
        use_dist_loss = False
        lens1 = [len(x) for x in self.accumulate_for_loss]
        if np.sum(lens1) >= self.acc_amount:
            use_dist_loss = True
        if use_dist_loss:
            total_amount = 0
            dist_losses = [0] * len(self.accumulate_for_loss)
            for i, features in enumerate(self.accumulate_for_loss):
                if len(features) > 0:
                    curr_amount = len(features)
                    total_amount += curr_amount
                    features = torch.mean(torch.stack(features), dim=0)
                    dist_losses[i] = torch.mean((features - self.best_matchs[i].to(features.device)) ** 2) * curr_amount
                    self.accumulate_for_loss[i] = []
            for l in dist_losses:
                if l > 0:
                    dist_loss += l
            dist_loss /= total_amount
            if float(dist_loss) > 0:
                self.avg_dist_losses.append(float(dist_loss))
        wandb.log({'avg_dist_losses': float(np.mean(self.avg_dist_losses)), 'acc': float(np.sum(lens1)),
                   'source_feats': len(self.slice_to_feature_source),
                   'target_feats': len(self.slice_to_feature_target)})
        return dist_loss, {'avg_dist_losses': np.mean(self.avg_dist_losses)}

    # with torch.no_grad():
    #     self.get_imnet_model().eval()
    #     feat_imnet = self.get_imnet_model().extract_feat(img)
    #     feat_imnet = [f.detach() for f in feat_imnet]
    # lay = -1
    # if self.fdist_classes is not None:
    #     fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
    #     scale_factor = gt.shape[-1] // feat[lay].shape[-1]
    #     gt_rescaled = downscale_label_ratio(gt, scale_factor,
    #                                         self.fdist_scale_min_ratio,
    #                                         self.num_classes,
    #                                         255).long().detach()
    #     fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
    #     feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
    #                                       fdist_mask)
    #     self.debug_fdist_mask = fdist_mask
    #     self.debug_gt_rescale = gt_rescaled
    # else:
    #     feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
    # feat_dist = self.fdist_lambda * feat_dist
    # feat_loss, feat_log = self._parse_losses(
    #     {'loss_imnet_feat_dist': feat_dist})
    # feat_log.pop('loss', None)
    # return feat_loss, feat_log
