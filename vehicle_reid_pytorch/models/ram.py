from vehicle_reid_pytorch.models import Baseline
from torch import nn
from .baseline import weights_init_classifier, weights_init_kaiming


class RAM(Baseline):
    def __init__(self, divides, *args, **kwargs):
        """RAM
        
        Arguments:
            Baseline {[type]} -- [description]
            divide {List[(int, int, int, int, int, int)]} -- 如[(0, 1024, 0, 6, 0, 16), (4, 8, 0, 16), (8, 16, 0, 16)]，
            数组内容用于切片。左闭右开。如果是6个值则在通道上做切分，如果是4个值则仅在高宽做切分。
        """
        super(RAM, self).__init__(*args, **kwargs)
        self.divides = divides
        self.num_parts = len(self.divides)
        self.local_bottlenecks = nn.ModuleList([])
        self.local_classifiers = nn.ModuleList([])
        for i, divide in enumerate(divides):
            if len(divide) == 6:
                channels = divide[1] - divide[0]
                print(channels)
            else:
                channels = self.in_planes
            local_bottleneck = nn.BatchNorm1d(channels)
            local_classifier = nn.Linear(channels, self.num_classes, bias=False)
            local_bottleneck.bias.require_grads = False
            local_bottleneck.apply(weights_init_kaiming)
            local_classifier.apply(weights_init_classifier)
            self.local_bottlenecks.append(local_bottleneck)
            self.local_classifiers.append(local_classifier)
        

    def get_feature(self, feature_map, classifier, bottleneck):
        global_feat = self.gap(feature_map)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def forward(self, x, **kwargs):
        x = self.base(x)
        if self.training:
            global_score, global_feat = self.get_feature(x, self.classifier, self.bottleneck)  # (b, 2048, 1, 1)
            local_feats = []
            local_scores = []
            for i, divide in enumerate(self.divides):
                if len(divide) == 6:
                    lc, rc, lh, rh, lw, rw = divide
                elif len(divide) == 4:
                    lh, rh, lw, rw = divide
                    lc, rc = 0, self.in_planes

                local_fmap = x[:, lc:rc, lh:rh, lw:rw] 
                local_score, local_feat = self.get_feature(local_fmap, self.local_classifiers[i], self.local_bottlenecks[i])
                local_feats.append(local_feat)
                local_scores.append(local_score)
            return {
                "local_feats": local_feats,
                "local_scores": local_scores,
                "global_feat": global_feat,
                "global_score": global_score
            }
        else:
            global_feat = self.get_feature(x, self.classifier, self.bottleneck)  # (b, 2048, 1, 1)
            # local_feats = []
            # for i, divide in enumerate(self.divides):
            #     local_feat = self.get_feature(x, self.local_classifiers[i], self.local_bottlenecks[i])
            #     local_feats.append(local_feat)
            return {
                # "local_feats": local_feats,
                "global_feat": global_feat,
            }



